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
	"github.com/hashicorp/terraform/internal/plans/planfile"
	"github.com/hashicorp/terraform/internal/tfdiags"
)

// ApplyCommand is a Command implementation that applies a Terraform
// configuration and actually builds or changes infrastructure.
type ApplyCommand struct {
	Meta

	// If true, then this apply command will become the "destroy"
	// command. It is just like apply but only processes a destroy.
	Destroy bool
}

func (c *ApplyCommand) Run(rawArgs []string) int {
	var diags tfdiags.Diagnostics

	// Parse and apply global view arguments
	common, rawArgs := arguments.ParseView(rawArgs)
	c.View.Configure(common)

	// Propagate -no-color for legacy use of Ui.  The remote backend and
	// cloud package use this; it should be removed when/if they are
	// migrated to views.
	c.Meta.color = !common.NoColor
	c.Meta.Color = c.Meta.color

	// Parse and validate flags
	var args *arguments.Apply
	switch {
	case c.Destroy:
		args, diags = arguments.ParseApplyDestroy(rawArgs)
	default:
		args, diags = arguments.ParseApply(rawArgs)
	}
	diags = diags.Append(c.Validate(args))

	// Instantiate the view, even if there are flag errors, so that we render
	// diagnostics according to the desired view
	view := views.NewApply(args.ViewType, c.Destroy, c.View)

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

	// Attempt to load the plan file, if specified
	planFile, loadPlanFileDiags := c.LoadPlanFile(args.PlanPath)
	diags = diags.Append(loadPlanFileDiags)
	if diags.HasErrors() {
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

	// Prepare the backend, passing the plan file if present, and the
	// backend-specific arguments
	be, beDiags := c.PrepareBackend(planFile, args.State, args.ViewType)
	diags = diags.Append(beDiags)
	if diags.HasErrors() {
		view.Diagnostics(diags)
		return 1
	}

	// Build the operation request
	opReq, opDiags := c.OperationRequest(be, view, args.ViewType, planFile, args.Operation, args.AutoApprove, args.PolicyPaths)
	diags = diags.Append(opDiags)
	if diags.HasErrors() {
		view.Diagnostics(diags)
		return 1
	}

	if len(args.PolicyPaths) > 0 {
		client, policyDiags, stopClient := c.PolicyClient(context.Background(), args.PolicyPaths)
		// if there has been any errors when setting up the policy client, we log them but
		// we still proceed with the operation, as a failure to set up the policy client
		// should not prevent the apply operation from running
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

	// Before we delegate to the backen