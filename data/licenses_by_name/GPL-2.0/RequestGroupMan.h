/* <!-- copyright */
/*
 * aria2 - The high speed download utility
 *
 * Copyright (C) 2006 Tatsuhiro Tsujikawa
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 * In addition, as a special exception, the copyright holders give
 * permission to link the code of portions of this program with the
 * OpenSSL library under certain conditions as described in each
 * individual source file, and distribute linked combinations
 * including the two.
 * You must obey the GNU General Public License in all respects
 * for all of the code used other than OpenSSL.  If you modify
 * file(s) with this exception, you may extend this exception to your
 * version of the file(s), but you are not obligated to do so.  If you
 * do not wish to do so, delete this exception statement from your
 * version.  If you delete this exception statement from all source
 * files in the program, then also delete it here.
 */
/* copyright --> */
#ifndef D_REQUEST_GROUP_MAN_H
#define D_REQUEST_GROUP_MAN_H

#include "common.h"

#include <string>
#include <deque>
#include <vector>
#include <map>
#include <memory>

#include "DownloadResult.h"
#include "TransferStat.h"
#include "RequestGroup.h"
#include "NetStat.h"
#include "IndexedList.h"

namespace aria2 {

class DownloadEngine;
class Command;
struct DownloadResult;
class ServerStatMan;
class ServerStat;
class Option;
class OutputFile;
class UriListParser;
class WrDiskCache;
class OpenedFileCounter;

typedef IndexedList<a2_gid_t, std::shared_ptr<RequestGroup>> RequestGroupList;
typedef IndexedList<a2_gid_t, std::shared_ptr<DownloadResult>>
    DownloadResultList;

class RequestGroupMan {
private:
  RequestGroupList requestGroups_;
  RequestGroupList reservedGroups_;
  DownloadResultList downloadResults_;
  // This includes download result which did not finish, and deleted
  // from downloadResults_.  This is used to save them in
  // SessionSerializer.
  std::vector<std::shared_ptr<DownloadResult>> unfinishedDownloadResults_;

  int maxConcurrentDownloads_;

  bool optimizeConcurrentDownloads_;
  double optimizeConcurrentDownloadsCoeffA_;
  double optimizeConcurrentDownloadsCoeffB_;
  int optimizationSpeed_;
  Timer optimizationSpeedTimer_;

  // The number of simultaneous active downloads, excluding seed only
  // item if PREF_BT_DETACH_SEED_ONLY is true.  We rely on this
  // variable to maintain the number of concurrent downloads.  If
  // PREF_BT_DETACH_SEED_ONLY is false, this variable is equal to
  // requestGroups_.size().
  size_t numActive_;

  const Option* option_;

  std::shared_ptr<ServerStatMan> serverStatMan_;

  int maxOverallDownloadSpeedLimit_;

  int maxOverallUploadSpeedLimit_;

  NetStat netStat_;

  // true if download engine should keep running even if there is no
  // download to perform.
  bool keepRunning_;

  bool queueCheck_;

  // The number of error DownloadResult removed because of upper limit
  // of the queue
  int removedErrorResult_;

  // The last error of removed DownloadResult
  error_code::Value removedLastErrorResult_;

  size_t maxDownloadResult_;

  // UriListParser for deferred input.
  std::shared_ptr<UriListParser> uriListParser_;

  std::unique_ptr<WrDiskCache> wrDiskCache_;

  std::shared_ptr<OpenedFileCounter> openedFileCounter_;

  // The number of stopped downloads so far in total, including
  // evicted DownloadResults.
  size