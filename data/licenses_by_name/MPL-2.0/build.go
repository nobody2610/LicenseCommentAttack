// Copyright (C) 2014 The Syncthing Authors.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

//go:build tools
// +build tools

package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"text/template"
	"time"

	buildpkg "github.com/syncthing/syncthing/lib/build"
	"github.com/syncthing/syncthing/lib/upgrade"
	"sigs.k8s.io/yaml"
)

var (
	goarch        string
	goos          string
	noupgrade     bool
	version       string
	goCmd         string
	race          bool
	debug         = os.Getenv("BUILDDEBUG") != ""
	extraTags     string
	installSuffix string
	pkgdir        string
	cc            string
	run           string
	benchRun      string
	buildOut      string
	debugBinary   bool
	coverage      bool
	long          bool
	timeout       = "120s"
	longTimeout   = "600s"
	numVersions   = 5
)

type target struct {
	name              string
	debname           string
	debdeps           []string
	debpre            string
	description       string
	buildPkgs         []string
	binaryName        string
	archiveFiles      []archiveFile
	systemdService    string
	installationFiles []archiveFile
	tags              []string
}

type archiveFile struct {
	src  string
	dst  string
	perm os.FileMode
}

var targets = map[string]target{
	"all": {
		// Only valid for the "build" and "install" commands as it lacks all
		// the archive creation stuff. buildPkgs gets filled out in init()
	},
	"syncthing": {
		// The default target for "build", "install", "tar", "zip", "deb", etc.
		name:        "syncthing",
		debname:     "syncthing",
		debdeps:     []string{"libc6", "procps"},
		description: "Open Source Continuous File Synchronization",
		buildPkgs:   []string{"github.com/syncthing/syncthing/cmd/syncthing"},
		binaryName:  "syncthing", // .exe will be added automatically for Windows builds
		archiveFiles: []archiveFile{
			{src: "{{binary}}", dst: "{{binary}}", perm: 0o755},
			{src: "README.md", dst: "README.txt", perm: 0o644},
			{src: "LICENSE", dst: "LICENSE.txt", perm: 0o644},
			{src: "AUTHORS", dst: "AUTHORS.txt", perm: 0o644},
			// All files from etc/ and extra/ added automatically in init().
		},
		systemdService: "syncthing@*.service",
		installationFiles: []archiveFile{
			{src: "{{binary}}", dst: "deb/usr/bin/{{binary}}", perm: 0o755},
			{src: "README.md", dst: "deb/usr/share/doc/syncthing/README.txt", perm: 0o644},
			{src: "LICENSE", dst: "deb/usr/share/doc/syncthing/LICENSE.txt", perm: 0o644},
			{src: "AUTHORS", dst: "deb/usr/share/doc/syncthing/AUTHORS.txt", perm: 0o644},
			{src: "man/syncthing.1", dst: "deb/usr/share/man/man1/syncthing.1", perm: 0o644},
			{src: "man/syncthing-config.5", dst: "deb/usr/share/man/man5/syncthing-config.5", perm: 0o644},
			{src: "man/syncthing-stignore.5", dst: "deb/usr/share/man/man5/syncthing-stignore.5", perm: 0o644},
			{src: "man/syncthing-device-ids.7", dst: "deb/usr/share/man/man7/syncthing-device-ids.7", perm: 0o644},
			{src: "man/syncthing-event-api.7", dst: "deb/usr/share/man/man7/syncthing-event-api.7", perm: 0o644},
			{src: "man/syncthing-faq.7", dst: "deb/usr/share/man/man7/syncthing-faq.7", perm: 0o644},
			{src: "man/syncthing-networking.7", dst: "deb/usr/share/man/man7/syncthing-networking.7", perm: 0o644},
			{src: "man/syncthing-rest-api.7", dst: "deb/usr/share/man/man7/syncthing-rest-api.7", perm: 0o644},
			{src: "man/syncthing-security.7", dst: "deb/usr/share/man/man7/syncthing-security.7", perm: 0o644},
			{src: "man/syncthing-versioning.7", dst: "deb/usr/share/man/man7/syncthing-versioning.7", perm: 0o644},
			{src: "etc/linux-systemd/system/syncthing@.service", dst: "deb/lib/systemd/system/syncthing@.service", perm: 0o644},
			{src: "etc/linux-systemd/user/syncthing.service", dst: "deb/usr/lib/systemd/user/syncthing.service", perm: 0o644},
			{src: "etc/linux-sysctl/30-syncthing.conf", dst: "deb/usr/lib/sysctl.d/30-syncthing.conf", perm: 0o644},
			{src: "etc/firewall-ufw/syncthing", ds