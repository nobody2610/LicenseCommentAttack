// Copyright IBM Corp. 2016, 2025
// SPDX-License-Identifier: BUSL-1.1


package http

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"mime"
	"net"
	"net/http"
	"net/http/pprof"
	"net/textproto"
	"net/url"
	"os"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/hashicorp/errwrap"
	"github.com/hashicorp/go-cleanhttp"
	"github.com/hashicorp/go-secure-stdlib/parseutil"
	"github.com/hashicorp/go-sockaddr"
	"github.com/hashicorp/go-uuid"
	"github.com/hashicorp/vault/helper/namespace"
	"github.com/hashicorp/vault/http/priority"
	"github.com/hashicorp/vault/internalshared/configutil"
	"github.com/hashicorp/vault/limits"
	"github.com/hashicorp/vault/sdk/helper/consts"
	"github.com/hashicorp/vault/sdk/helper/jsonutil"
	"github.com/hashicorp/vault/sdk/helper/pathmanager"
	"github.com/hashicorp/vault/sdk/logical"
	"github.com/hashicorp/vault/vault"
	gziphandler "github.com/klauspost/compress/gzhttp"
)

const (
	// WrapTTLHeaderName is the name of the header containing a directive to
	// wrap the response
	WrapTTLHeaderName = "X-Vault-Wrap-TTL"

	// WrapFormatHeaderName is the name of the header containing the format to
	// wrap in; has no effect if the wrap TTL is not set
	WrapFormatHeaderName = "X-Vault-Wrap-Format"

	// NoRequestForwardingHeaderName is the name of the header telling Vault
	// not to use request forwarding
	NoRequestForwardingHeaderName = "X-Vault-No-Request-Forwarding"

	// MFAHeaderName represents the HTTP header which carries the credentials
	// required to perform MFA on any path.
	MFAHeaderName = "X-Vault-MFA"

	// canonicalMFAHeaderName is the MFA header value's format in the request
	// headers. Do not alter the casing of this string.
	canonicalMFAHeaderName = "X-Vault-Mfa"

	// PolicyOverrideHeaderName is the header set to request overriding
	// soft-mandatory Sentinel policies.
	PolicyOverrideHeaderName = "X-Vault-Policy-Override"

	VaultIndexHeaderName        = "X-Vault-Index"
	VaultInconsistentHeaderName = "X-Vault-Inconsistent"
	VaultForwardHeaderName      = "X-Vault-Forward"
	VaultInconsistentForward    = "forward-active-node"
	VaultInconsistentFail       = "fail"

	// DefaultMaxRequestSize is the default maximum accepted request size. This
	// is to prevent a denial of service attack where no Content-Length is
	// provided and the server is fed ever more data until it exhausts memory.
	// Can be overridden per listener.
	DefaultMaxRequestSize = 32 * 1024 * 1024

	// VaultSnapshotReadParam is the query parameter sent when Vault should read
	// the data from a loaded snapshot
	VaultSnapshotReadParam = "read_snapshot_id"
	// VaultSnapshotRecoverParam is the query parameter sent when Vault should
	// recover the data from a loaded snapshot
	// Deprecated: use VaultSnapshotRecoverHeader
	VaultSnapshotRecoverParam = "recover_snapshot_id"
	// VaultRecoverSourcePathHeader contains an optional source path
	// to read the data from when performing a recover operation
	VaultRecoverSourcePathHeader = consts.RecoverSourcePathHeader
	// VaultSnapshotRecoverHeader holds the snapshot ID to use for a read, list, or
	// recover from snapshot operation. This replaces the use of query parameters
	// to pass the snapshot ID
	VaultSnapshotRecoverHeader = "X-Vault-Recover-Snapshot-Id"

	// DefaultMaxTokenHeaderSize is the default maximum size in bytes for an
	// authentication token passed in the X-Vault-Token and Authorization: Bearer
	// headers. This is to prevent a denial of service attack via unbounded
	// header values. Can be overridden per listener.
	DefaultMaxTokenHeaderSize = 8 * 1024 // 8 KB

	// CustomMaxJSONDepth specifies the maximum nesting depth of a JSON object.
	// This limit is designed to prevent stack exhaustion attacks from deeply
	// nested JSON payloads, which could otherwise lead to a denial-of-service
	// (DoS) vulnerability. The default value of 300 is intentionally generous
	// to support complex but legitimate configurations, while still providing
	// a safeguard agai