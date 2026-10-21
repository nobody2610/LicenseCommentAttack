// Copyright (c) 2015-2024 MinIO, Inc.
//
// This file is part of MinIO Object Storage stack
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


package cmd

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/coreos/go-systemd/v22/daemon"
	"github.com/dustin/go-humanize"
	"github.com/minio/cli"
	"github.com/minio/madmin-go/v3"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/minio/minio-go/v7/pkg/set"
	"github.com/minio/minio/internal/auth"
	"github.com/minio/minio/internal/bucket/bandwidth"
	"github.com/minio/minio/internal/color"
	"github.com/minio/minio/internal/config"
	"github.com/minio/minio/internal/config/api"
	"github.com/minio/minio/internal/handlers"
	"github.com/minio/minio/internal/hash/sha256"
	xhttp "github.com/minio/minio/internal/http"
	xioutil "github.com/minio/minio/internal/ioutil"
	"github.com/minio/minio/internal/logger"
	"github.com/minio/pkg/v3/certs"
	"github.com/minio/pkg/v3/env"
	"gopkg.in/yaml.v2"
)

// ServerFlags - server command specific flags
var ServerFlags = []cli.Flag{
	cli.StringFlag{
		Name:   "config",
		Usage:  "specify server configuration via YAML configuration",
		EnvVar: "MINIO_CONFIG",
	},
	cli.StringFlag{
		Name:   "address",
		Value:  ":" + GlobalMinioDefaultPort,
		Usage:  "bind to a specific ADDRESS:PORT, ADDRESS can be an IP or hostname",
		EnvVar: "MINIO_ADDRESS",
	},
	cli.IntFlag{
		Name:   "listeners", // Deprecated Oct 2022
		Value:  1,
		Usage:  "bind N number of listeners per ADDRESS:PORT",
		EnvVar: "MINIO_LISTENERS",
		Hidden: true,
	},
	cli.StringFlag{
		Name:   "console-address",
		Usage:  "bind to a specific ADDRESS:PORT for embedded Console UI, ADDRESS can be an IP or hostname",
		EnvVar: "MINIO_CONSOLE_ADDRESS",
	},
	cli.DurationFlag{
		Name:   "shutdown-timeout",
		Value:  time.Second * 30,
		Usage:  "shutdown timeout to gracefully shutdown server (DEPRECATED)",
		EnvVar: "MINIO_SHUTDOWN_TIMEOUT",
		Hidden: true,
	},

	cli.DurationFlag{
		Name:   "idle-timeout",
		Value:  xhttp.DefaultIdleTimeout,
		Usage:  "idle timeout is the maximum amount of time to wait for the next request when keep-alive are enabled",
		EnvVar: "MINIO_IDLE_TIMEOUT",
		Hidden: true,
	},
	cli.DurationFlag{
		Name:   "read-header-timeout",
		Value:  xhttp.DefaultReadHeaderTimeout,
		Usage:  "read header timeout is the amount of time allowed to read request headers",
		EnvVar: "MINIO_READ_HEADER_TIMEOUT",
		Hidden: true,
	},
	cli.DurationFlag{
		Name:   "conn-user-timeout",
		Usage:  "custom TCP_USER_TIMEOUT for socket buffers",
		Hidden: true,
		Value:  10 * time.Minute,
		EnvVar: "MINIO_CONN_USER_TIMEOUT",
	},
	cli.StringFlag{
		Name:   "interface",
		Usage:  "bind to right VRF device for MinIO services",
		Hidden: true,
		EnvVar: "MINIO_INTERFACE",
	},
	cli.DurationFlag{
		Name:   "dns-cache-ttl",
		Usage:  "custom DNS cache TTL",
		Hidden: true,
		Value: func() time.Duration {
			if orchestrated {
				return 30 * time.Second
			}
			return 10 * time.Minute
		}(),
		EnvVar: "MINIO_DNS_CACHE_TTL",
	},
	cli.IntFlag{
		Name:   "max-idle-conns-per-host",
		Usage:  "set a custom max idle connections per host value",
		Hidden: true,
		Value:  2048,
		EnvVar: "MINIO_MAX_IDLE_CONNS_PER_HOST",
	},
	cli.StringSliceFlag{
		Name:  "ftp",
		Usage: "enable and configure an FTP(Secure) server",
	},
	cli.StringSliceFlag{
		Name:  "sftp",
		Usage: "enable and configure an SFTP server",
	},
	cli.StringFlag{
		Name:   "crossdomain-xml",
		Usage:  "provide a custom crossdomain-xml configuration to report at http://endpoint/crossdomain.xml",
		Hidden: true,
		EnvVar: "MINIO_CROSSDOMAIN_XML",
	},
	cli.StringFlag{
		Name:   "memlimit",
		Usage:  "set global memory limit per server via GOMEMLIMIT",
		Hidden: true,
		EnvVar: "MINIO_MEMLIMIT",
	},
	cli.IntFlag{
		Name:   "send-buf-size",
		Value:  4 * humanize.MiByte,
		EnvVar: "MINIO_SEND_BUF_SIZE",
		Hidden: true,
	},
	cli.IntFlag{
		Name:   "recv-buf-size",
		Value:  4 * humanize.MiByte,
		EnvVar: "MINIO_RECV_BUF_SIZE",