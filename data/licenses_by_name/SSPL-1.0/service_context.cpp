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



#include "mongo/db/service_context.h"

#include "mongo/base/init.h"  // IWYU pragma: keep
#include "mongo/base/initializer.h"
#include "mongo/db/client.h"
#include "mongo/db/default_baton.h"
#include "mongo/db/op_observer/op_observer.h"
#include "mongo/db/operation_context.h"
#include "mongo/db/server_options.h"
#include "mongo/db/storage/recovery_unit.h"
#include "mongo/db/storage/recovery_unit_noop.h"
#include "mongo/db/storage/write_unit_of_work.h"
#include "mongo/logv2/log.h"
#include "mongo/transport/service_entry_point.h"
#include "mongo/transport/session.h"
#include "mongo/transport/transport_layer_manager.h"
#include "mongo/util/assert_util.h"
#include "mongo/util/fail_point.h"
#include "mongo/util/observable_mutex_registry.h"
#include "mongo/util/scopeguard.h"
#include "mongo/util/system_clock_source.h"
#include "mongo/util/system_tick_source.h"

#include <exception>
#include <list>
#include <memory>

#include <absl/container/node_hash_set.h>
#include <absl/meta/type_traits.h>
#include <boost/optional/optional.hpp>

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kDefault


namespace mongo {
namespace {

ServiceContext* globalServiceContext = nullptr;

MONGO_FAIL_POINT_DEFINE(hangBeforeNotifyStorageStartupRecoveryComplete);
MONGO_FAIL_POINT_DEFINE(releaseFailPointAfterGlobalKillSet);

}  // namespace

ClientLock::ClientLock(Client* client) : service_context_detail::ObjectLock<Client>(client) {}

bool hasGlobalServiceContext() {
    return globalServiceContext;
}

ServiceContext* getGlobalServiceContext() {
    fassert(17508, globalServiceContext);
    return globalServiceContext;
}

ServiceContext* getCurrentServiceContext() {
    auto client = Client::getCurrent();
    if (client) {
        return client->getServiceContext();
    }

    return nullptr;
}

void setGlobalServiceContext(ServiceContext::UniqueServiceContext&& serviceContext) {
    if (globalServiceContext) {
        // Make sure that calling getGlobalServiceContext() during the destructor results in
        // nullptr. Decorations might try and do this.
        ServiceContext::UniqueServiceContext oldServiceContext{globalServiceContext};
        globalServiceContext = nullptr;
    }

    globalServiceContext = serviceContext.release();
}

logv2::LogService toLogService(Service* service) {
    return toLogService(service ? service->role() : ClusterRole::None);
}

/**
 * The global clusterRole determines which services are initialized.
 * If no role is set, then ShardServer is assumed, so there's always
 * at least one Service created.
 */
struct ServiceContext::ServiceSet {
public:
    explicit ServiceSet(ServiceContext* sc) {
        auto role = serverGlobalParams.clusterRole;
        if (!role.has(ClusterRole::RouterServer))
            role += ClusterRole::ShardServer;
        if (role.has(ClusterRole::RouterServer))
            _router = Service::make(sc, ClusterRole::RouterServer);
        if (role.has(ClusterRole::ShardServer))
            _shard = Service::make(sc, ClusterRole::ShardServer);
    }

    /** The `role` here must be ShardServer or RouterServer exactly. */
    Service* getService(ClusterRole role) {
        if (role.hasExclusively(ClusterRole::ShardServer))
            return _shard.get();
        if (role.hasExclusively(ClusterRole::RouterServer))
            return _router.get();
        MONGO_UNREACHABLE;
    }

private:
    Service::UniqueService _shard;
    Service::UniqueService _router;
};

Service::~Service() = default;
Service::Service(ServiceContext* sc, ClusterRole role) : _sc{sc}, _role{role} {}

ServiceContext::ServiceContext(std::unique_ptr<ClockSource> fastClockSource,
                               std::unique_ptr<ClockSource> preciseClockSource,
                               std::unique_ptr<TickSource> tickSource)
    : _tickSource(std::move(tickSource)),
      _fastClockSource(std::move(fastClockSource)),
      _preciseClockSource(std::move(preciseClockSource)),
      _serviceSet(std::make_uniqu