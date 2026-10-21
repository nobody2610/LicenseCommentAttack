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



#include <cstring>

#include <s2cellid.h>

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional.hpp>
#include <boost/smart_ptr/intrusive_ptr.hpp>
// IWYU pragma: no_include "ext/alloc_traits.h"
#include "mongo/base/checked_cast.h"
#include "mongo/base/error_codes.h"
#include "mongo/base/string_data.h"
#include "mongo/bson/bsonelement.h"
#include "mongo/bson/bsonobj.h"
#include "mongo/bson/bsonobjbuilder.h"
#include "mongo/bson/bsontypes.h"
#include "mongo/db/commands/server_status/server_status_metric.h"
#include "mongo/db/index/index_constants.h"
#include "mongo/db/index_names.h"
#include "mongo/db/matcher/expression_algo.h"
#include "mongo/db/matcher/expression_hasher.h"
#include "mongo/db/matcher/expression_text.h"
#include "mongo/db/pipeline/document_source.h"
#include "mongo/db/pipeline/document_source_group.h"
#include "mongo/db/pipeline/document_source_internal_projection.h"
#include "mongo/db/pipeline/document_source_internal_replace_root.h"
#include "mongo/db/pipeline/document_source_internal_unpack_bucket.h"
#include "mongo/db/pipeline/document_source_lookup.h"
#include "mongo/db/pipeline/document_source_set_window_fields.h"
#include "mongo/db/pipeline/document_source_skip.h"
#include "mongo/db/pipeline/expression_context.h"
#include "mongo/db/pipeline/field_path.h"
#include "mongo/db/pipeline/search/search_helper.h"
#include "mongo/db/query/canonical_query.h"
#include "mongo/db/query/collation/collation_index_key.h"
#include "mongo/db/query/collation/collator_interface.h"
#include "mongo/db/query/compiler/ce/exact/exact_cardinality.h"
#include "mongo/db/query/compiler/logical_model/projection/projection.h"
#include "mongo/db/query/compiler/logical_model/sort_pattern/sort_pattern.h"
#include "mongo/db/query/compiler/metadata/index_entry.h"
#include "mongo/db/query/compiler/optimizer/cost_based_ranker/cardinality_estimator.h"
#include "mongo/db/query/compiler/optimizer/cost_based_ranker/cost_estimator.h"
#include "mongo/db/query/compiler/physical_model/query_solution/eof_node_type.h"
#include "mongo/db/query/compiler/physical_model/query_solution/query_solution.h"
#include "mongo/db/query/compiler/physical_model/query_solution/stage_types.h"
#include "mongo/db/query/distinct_access.h"
#include "mongo/db/query/find_command.h"
#include "mongo/db/query/index_tag.h"
#include "mongo/db/query/plan_cache/classic_plan_cache.h"
#include "mongo/db/query/plan_cache/plan_cache_diagnostic_printer.h"
#include "mongo/db/query/plan_enumerator/plan_enumerator.h"
#include "mongo/db/query/plan_enumerator/plan_enumerator_explain_info.h"
#include "mongo/db/query/plan_ranking/plan_ranker.h"
#include "mongo/db/query/planner_access.h"
#include "mongo/db/query/planner_analysis.h"
#include "mongo/db/query/planner_ixselect.h"
#include "mongo/db/query/query_execution_knobs_gen.h"
#include "mongo/db/query/query_knobs/query_knob_configuration.h"
#include "mongo/db/query/query_optimization_knobs_gen.h"
#include "mongo/db/query/query_planner.h"
#include "mongo/db/query/query_planner_common.h"
#include "mongo/db/query/query_request_helper.h"
#include "mongo/db/query/search/mongot_cursor.h"
#include "mongo/db/shard_role/shard_catalog/clustered_collection_options_gen.h"
#include "mongo/logv2/log.h"
#include "mongo/util/assert_util.h"
#include "mongo/util/str.h"

#include <deque>
#include <string>
#include <utility>
#include <vector>

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kQuery

namespace mongo {

namespace {
/**
 * Aggregation of the total number of candidate plans that couldn't be costed in CBR.
 */
auto& numPlansFailedCostEstimation =
    *MetricBuilder<Counter64>{"query.cbr.numPlansFailedCostEstimation"};

/**
 * Aggregation of the total number of times candidate plans tied in cost estimation in CBR.
 */
auto& numPlansTiedCostEstimation =
    *MetricBuilder<Counter64>{"query.cbr.numPlansTiedCostEstimation"};

/**
 * Aggregation of the total number of times CBR successfully chose a