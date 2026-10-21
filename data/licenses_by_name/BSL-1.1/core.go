// Copyright IBM Corp. 2016, 2025
// SPDX-License-Identifier: BUSL-1.1


package vault

import (
	"context"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	paths "path"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/errwrap"
	log "github.com/hashicorp/go-hclog"
	wrapping "github.com/hashicorp/go-kms-wrapping/v2"
	aeadwrapper "github.com/hashicorp/go-kms-wrapping/wrappers/aead/v2"
	"github.com/hashicorp/go-kms-wrapping/wrappers/awskms/v2"
	"github.com/hashicorp/go-multierror"
	"github.com/hashicorp/go-secure-stdlib/mlock"
	"github.com/hashicorp/go-secure-stdlib/reloadutil"
	"github.com/hashicorp/go-secure-stdlib/strutil"
	"github.com/hashicorp/go-secure-stdlib/tlsutil"
	"github.com/hashicorp/go-uuid"
	lru "github.com/hashicorp/golang-lru/v2"
	kv "github.com/hashicorp/vault-plugin-secrets-kv"
	"github.com/hashicorp/vault/api"
	"github.com/hashicorp/vault/audit"
	"github.com/hashicorp/vault/command/server"
	"github.com/hashicorp/vault/helper/activationflags"
	"github.com/hashicorp/vault/helper/identity/mfa"
	"github.com/hashicorp/vault/helper/locking"
	"github.com/hashicorp/vault/helper/metricsutil"
	"github.com/hashicorp/vault/helper/namespace"
	"github.com/hashicorp/vault/helper/osutil"
	"github.com/hashicorp/vault/helper/trace"
	"github.com/hashicorp/vault/physical/raft"
	"github.com/hashicorp/vault/sdk/helper/certutil"
	"github.com/hashicorp/vault/sdk/helper/consts"
	"github.com/hashicorp/vault/sdk/helper/jsonutil"
	"github.com/hashicorp/vault/sdk/helper/logging"
	"github.com/hashicorp/vault/sdk/helper/pathmanager"
	"github.com/hashicorp/vault/sdk/logical"
	"github.com/hashicorp/vault/sdk/physical"
	sr "github.com/hashicorp/vault/serviceregistration"
	"github.com/hashicorp/vault/shamir"
	"github.com/hashicorp/vault/vault/billing"
	"github.com/hashicorp/vault/vault/cert_count"
	"github.com/hashicorp/vault/vault/cluster"
	"github.com/hashicorp/vault/vault/eventbus"
	"github.com/hashicorp/vault/vault/observations"
	"github.com/hashicorp/vault/vault/plugincatalog"
	"github.com/hashicorp/vault/vault/quotas"
	vaultseal "github.com/hashicorp/vault/vault/seal"
	"github.com/hashicorp/vault/version"
	"github.com/patrickmn/go-cache"
	uberAtomic "go.uber.org/atomic"
	"google.golang.org/grpc"
)

const (
	// CoreLockPath is the path used to acquire a coordinating lock
	// for a highly-available deploy.
	CoreLockPath = "core/lock"

	// The poison pill is used as a check during certain scenarios to indicate
	// to standby nodes that they should seal
	poisonPillPath   = "core/poison-pill"
	poisonPillDRPath = "core/poison-pill-dr"

	// coreLeaderPrefix is the prefix used for the UUID that contains
	// the currently elected leader.
	coreLeaderPrefix = "core/leader/"

	// coreKeyringCanaryPath is used as a canary to indicate to replicated
	// clusters that they need to perform a rekey operation synchronously; this
	// isn't keyring-canary to avoid ignoring it when ignoring core/keyring
	coreKeyringCanaryPath = "core/canary-keyring"

	// coreGroupPolicyApplicationPath is used to store the behaviour for
	// how policies should be applied
	coreGroupPolicyApplicationPath = "core/group-policy-application-mode"

	// Path in storage for the plugin catalog.
	pluginCatalogPath = "core/plugin-catalog/"
	// Path in storage for the plugin runtime catalog.
	pluginRuntimeCatalogPath = "core/plugin-runtime-catalog/"

	// groupPolicyApplicationModeWithinNamespaceHierarchy is a configuration option for group
	// policy application modes, which allows only in-namespace-hierarchy policy application
	groupPolicyApplicationModeWithinNamespaceHierarchy = "within_namespace_hierarchy"

	// groupPolicyApplicationModeAny is a configuration option for group
	// policy application modes, which allows policy application irrespective of namespaces
	groupPolicyApplicationModeAny = "any"

	indexHeaderHMACKeyPath = "core/index-header-hmac-key"

	//