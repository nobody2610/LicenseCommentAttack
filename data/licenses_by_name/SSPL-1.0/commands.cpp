/**
 *    Copyright (C) 2018-present MongoDB, Inc.
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the Server Side Public License, version 1,
 *    as published by MongoDB, Inc.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    Server Side Public License for more details.
 *
 *    You should have received a copy of the Server Side Public License
 *    along with this program. If not, see
 *    <http://www.mongodb.com/licensing/server-side-public-license>.
 *
 *    As a special exception, the copyright holders give permission to link the
 *    code of portions of this program with the OpenSSL library under certain
 *    conditions as described in each individual source file and distribute
 *    linked combinations including the program with the OpenSSL library. You
 *    must comply with the Server Side Public License in all respects for
 *    all of the code used other than as permitted herein. If you modify file(s)
 *    with this exception, you may extend this exception to your version of the
 *    file(s), but you are not obligated to do so. If you do not wish to do so,
 *    delete this exception statement from your version. If you delete this
 *    exception statement from all source files in the program, then also delete
 *    it in the license file.
 */


#include "mongo/db/commands.h"

#include "mongo/base/error_extra_info.h"
#include "mongo/bson/bsontypes.h"
#include "mongo/bson/util/bson_extract.h"
#include "mongo/db/audit.h"
#include "mongo/db/auth/authorization_manager.h"
#include "mongo/db/auth/authorization_session.h"
#include "mongo/db/auth/validated_tenancy_scope.h"
#include "mongo/db/client.h"
#include "mongo/db/commands/test_commands_enabled.h"
#include "mongo/db/curop.h"
#include "mongo/db/curop_diagnostic_printer.h"
#include "mongo/db/error_labels.h"
#include "mongo/db/exec/mutable_bson/algorithm.h"
#include "mongo/db/exec/mutable_bson/document.h"
#include "mongo/db/generic_argument_util.h"
#include "mongo/db/namespace_string.h"
#include "mongo/db/namespace_string_util.h"
#include "mongo/db/service_context.h"
#include "mongo/db/tenant_id.h"
#include "mongo/db/topology/cluster_role.h"
#include "mongo/db/version_context.h"
#include "mongo/db/write_concern_options.h"
#include "mongo/idl/command_generic_argument.h"
#include "mongo/idl/idl_parser.h"
#include "mongo/logv2/log.h"
#include "mongo/platform/compiler.h"
#include "mongo/rpc/metadata/client_metadata.h"
#include "mongo/rpc/op_msg_rpc_impls.h"
#include "mongo/rpc/rewrite_state_change_errors.h"
#include "mongo/rpc/write_concern_error_detail.h"
#include "mongo/transport/session.h"
#include "mongo/util/assert_util.h"
#include "mongo/util/decorable.h"
#include "mongo/util/duration.h"
#include "mongo/util/fail_point.h"
#include "mongo/util/static_immortal.h"
#include "mongo/util/str.h"
#include "mongo/util/string_map.h"
#include "mongo/util/uuid.h"

#include <memory>
#include <string>
#include <vector>

#include <absl/container/node_hash_map.h>
#include <boost/move/utility_core.hpp>
#include <boost/none.hpp>
#include <boost/optional/optional.hpp>
#include <boost/smart_ptr.hpp>
#include <fmt/format.h>

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kCommand

namespace mongo {

const std::set<std::string> kNoApiVersions = {};
const std::set<std::string> kApiVersions1 = {"1"};

namespace {

const int kFailedFindCommandDebugLevel = 3;

const char kWriteConcernField[] = "writeConcern";

// Returns true if found to be authorized, false if undecided. Throws if unauthorized.
bool checkAuthorizationImplPreParse(
    OperationContext* opCtx,
    const Command* command,
    const boost::optional<auth::ValidatedTenancyScope>& validatedTenancyScope) {
    auto client = opCtx->getClient();
    if (client->isInDirectClient())
        return true;

    auto authzSession = AuthorizationSession::get(client);
    uassert(ErrorCodes::ReauthenticationRequired,
            fmt::format("Command {} requires reauthentication since the current authorization "
                        "session has expired. Please re-auth.",
                        command->getName()),
            !command->requiresAuth() || !authzSession->isExpired());

    if (!AuthorizationManager::get(opCtx->getService())->isAuthEnabled()) {
        // Running without auth, so everything should be allowed except remotely invoked
        // commands that have the 'localHostOnlyIfNoAuth' restriction.
        uassert(ErrorCodes::Unauthorized,
                str::stream() << command->getName()
                              << " must run from localhost when running db without auth",
                !command->adminOnly() || !command->localHostOnlyIfNoAuth() ||
                    client->getIsLocalHostConnection());
        return true;  // Blanket authorization: don't need to check anything else.
    }

    if (authzSession->isUsingLocalhostBypass())
        return false;  // Still can't decide on auth because of the localhost bypass.

    uassert(ErrorCodes::Unauthorized,
            str::stream() << "Command " << command->getName() << " requires authentication",
            !command->requiresAuth() || authzSession->isAuthenticated() ||
                (validatedTenancyScope && validatedTenancyScope->hasAuthenticatedUser()));

    return false;
}

void chec