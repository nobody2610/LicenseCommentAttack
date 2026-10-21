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



#include "mongo/db/storage/storage_engine_impl.h"

#include "mongo/base/error_codes.h"
#include "mongo/bson/bsonelement.h"
#include "mongo/bson/bsonobj.h"
#include "mongo/bson/timestamp.h"
#include "mongo/db/admission/execution_control/execution_admission_context.h"
#include "mongo/db/client.h"
#include "mongo/db/index/multikey_paths.h"
#include "mongo/db/operation_context.h"
#include "mongo/db/rss/replicated_storage_service.h"
#include "mongo/db/shard_role/shard_catalog/catalog_raii.h"
#include "mongo/db/shard_role/transaction_resources.h"
#include "mongo/db/storage/backup_cursor_hooks.h"
#include "mongo/db/storage/disk_space_monitor.h"
#include "mongo/db/storage/ident.h"
#include "mongo/db/storage/kv/kv_drop_pending_ident_reaper.h"
#include "mongo/db/storage/kv/kv_engine.h"
#include "mongo/db/storage/mdb_catalog.h"
#include "mongo/db/storage/record_data.h"
#include "mongo/db/storage/recovery_unit_noop.h"
#include "mongo/db/storage/spill_table.h"
#include "mongo/db/storage/storage_options.h"
#include "mongo/db/storage/storage_repair_observer.h"
#include "mongo/db/storage/write_unit_of_work.h"
#include "mongo/idl/idl_parser.h"
#include "mongo/logv2/log.h"
#include "mongo/logv2/log_component.h"
#include "mongo/logv2/log_severity.h"
#include "mongo/platform/compiler.h"
#include "mongo/stdx/unordered_set.h"
#include "mongo/util/assert_util.h"
#include "mongo/util/fail_point.h"
#include "mongo/util/log_and_backoff.h"
#include "mongo/util/str.h"

#include <algorithm>
#include <memory>

#ifdef _WIN32
#define NVALGRIND
#endif

#include <absl/container/node_hash_map.h>
#include <absl/meta/type_traits.h>
#include <boost/container/vector.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/none.hpp>
#include <boost/optional/optional.hpp>
#include <valgrind/valgrind.h>

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kStorage


#define LOGV2_FOR_RECOVERY(ID, DLEVEL, MESSAGE, ...) \
    LOGV2_DEBUG_OPTIONS(ID, DLEVEL, {logv2::LogComponent::kStorageRecovery}, MESSAGE, ##__VA_ARGS__)

namespace mongo {

using std::string;
using std::vector;

MONGO_FAIL_POINT_DEFINE(pauseTimestampMonitor);

namespace {
const auto kCatalogLogLevel = logv2::LogSeverity::Debug(2);

// Returns true if the ident refers to a resumable index build table.
bool isResumableIndexBuildIdent(StringData ident) {
    return ident::isInternalIdent(ident, kResumableIndexIdentStem);
}

// Idents corresponding to resumable index build tables are encoded as an 'internal' table and
// tagged 'kResumableIndexIdentStem'.
// Generates an ident to unique identify a new resumable index build table.
std::string generateNewResumableIndexBuildIdent() {
    const auto resumableIndexIdent = ident::generateNewInternalIdent(kResumableIndexIdentStem);
    invariant(isResumableIndexBuildIdent(resumableIndexIdent));
    return resumableIndexIdent;
}
}  // namespace

StorageEngineImpl::StorageEngineImpl(OperationContext* opCtx,
                                     std::unique_ptr<KVEngine> engine,
                                     std::unique_ptr<KVEngine> spillEngine,
                                     StorageEngineOptions options)
    : _engine(std::move(engine)),
      _spillEngine(std::move(spillEngine)),
      _options(std::move(options)),
      _dropPendingIdentReaper(_engine.get()),
      _timestampListener(
          [this](OperationContext* opCtx, const TimestampMonitor::Timestamps& timestamps) {
              _dropPendingIdentReaper.dropIdentsOlderThan(opCtx, timestamps);
          }),
      _supportsCappedCollections(_engine->supportsCappedCollections()) {

    // Replace the noop recovery unit for the startup operation context now that the storage engine
    // has been initialized. This is needed because at the time of startup, when the operation
    // context gets created, the storage engine initialization has not yet begun and so it gets
    // assigned a noop recovery unit. See the StorageClientObserver class.
    auto prevRecoveryUnit = shard_