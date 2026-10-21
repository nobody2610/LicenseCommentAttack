// Copyright IBM Corp. 2014, 2026
// SPDX-License-Identifier: BUSL-1.1


package command

import (
	"context"
	"fmt"
	"strings"

	"github.com/hashicorp/terraform/internal/backend/backendrun"
	"github.com/hashicorp/terraform/internal/command/arguments"
	"github.com/hashicorp/terraform/internal/command/views"
	"github.com/hashicorp/terraform/internal/tfdiags"
)

// PlanCommand is a Command implementation that compares a Terraform
// configuration to an actual infrastructure and shows the differences.
type PlanCommand struct {
	Meta
}

func (c *PlanCommand) Run(rawArgs []string) int {
	// Parse and apply global view arguments
	common, rawArgs := arguments.ParseView(rawArgs)
	c.View.Configure(common)

	// Propagate -no-color for legacy use of Ui.  The remote backend and
	// cloud package use this; it should be removed when/if they are
	// migrated to views.
	c.Meta.color = !common.NoColor
	c.Meta.Color = c.Meta.color

	// Parse and validate flags
	args, diags := arguments.ParsePlan(rawArgs)
	diags = diags.Append(c.Validate(args))

	// Instantiate the view, even if there are flag errors, so that we render
	// diagnostics according to the desired view
	view := views.NewPlan(args.ViewType, c.View)

	if diags.HasErrors() {
		view.Diagnostics(diags)
		view.HelpPrompt()
		return 1
	}

	// Check for user-supplied plugin path
	var err error
	if c.pluginPath, err = c.loadPluginPath(); err != nil {
		diags = diags.Append(err)
		view.Diagnostics(diags)
		return 1
	}

	// FIXME: the -input flag value is needed to initialize the backend and the
	// operation, but there is no clear path to pass this value down, so we
	// continue to mutate the Meta object state for now.
	c.Meta.input = args.InputEnabled

	// FIXME: the -parallelism flag is used to control the concurrency of
	// Terraform operations. At the moment, this value is used both to
	// initialize the backend via the ContextOpts field inside CLIOpts, and to
	// set a largely unused field on the Operation request. Again, there is no
	// clear path to pass this value down, so we continue to mutate the Meta
	// object state for now.
	c.Meta.parallelism = args.Operation.Parallelism

	// Prepare the backend with the backend-specific arguments
	be, beDiags := c.PrepareBackend(args.State, args.ViewType)
	b, isRemoteBackend := be.(BackendWithRemoteTerraformVersion)
	if isRemoteBackend && !b.IsLocalOperations() {
		diags = diags.Append(c.providerDevOverrideRuntimeWarningsRemoteExecution())
	} else {
		diags = diags.Append(c.providerDevOverrideRuntimeWarnings())
	}
	diags = diags.Append(beDiags)
	if diags.HasErrors() {
		view.Diagnostics(diags)
		return 1
	}

	// Build the operation request
	opReq, opDiags := c.OperationRequest(be, view, args.ViewType, args.Operation, args.OutPath, args.GenerateConfigPath, args.PolicyPaths)
	diags = diags.Append(opDiags)
	if diags.HasErrors() {
		view.Diagnostics(diags)
		return 1
	}

	if len(args.PolicyPaths) > 0 {
		client, policyDiags, stopClient := c.PolicyClient(context.Background(), args.PolicyPaths)
		// if there has been any errors when setting up the policy client, we log them but
		// we still proceed with the operation, as a failure to set up the policy client
		// should not prevent the plan operation from running
		if opReq.View != nil && policyDiags != nil {
			opReq.View.PolicyResults(nil, policyDiags)
		}
		opReq.PolicyClient = client
		defer stopClient()
	}

	// Collect variable value and add them to the operation request
	var varDiags tfdiags.Diagnostics
	opReq.Variables, varDiags = args.Vars.CollectValues(func(filename string, src []byte) {
		opReq.ConfigLoader.Parser().ForceFileSource(filename, src)
	})
	diags = diags.Append(varDiags)

	// Before we delegate to the backend, we'll print any warning diagnostics
	// we've accumulated here, since the backend will start fresh with its own
	// diagnostics.
	view.Diagnostics(diags)
	if diags.HasErrors() {
		return 1
	}
	diags = nil

	// Perform the operation
	op, err := c.RunOperation(be, opReq)
	if err != nil {
		diags = diags.Append(err)
		view.Diagnostics(diags)
		retu