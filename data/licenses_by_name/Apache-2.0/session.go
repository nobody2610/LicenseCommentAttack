/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package tensorflow

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"unsafe"
)

// Session drives a TensorFlow graph computation.
//
// When a Session is created with a given target, a new Session object is bound
// to the universe of resources specified by that target. Those resources are
// available to this session to perform computation described in the GraphDef.
// After creating the session with a graph, the caller uses the Run() API to
// perform the computation and potentially fetch outputs as Tensors.
// A Session allows concurrent calls to Run().
type Session struct {
	c *C.TF_Session

	// For ensuring that:
	// - Close() blocks on all Run() calls to complete.
	// - Close() can be called multiple times.
	wg sync.WaitGroup
	mu sync.Mutex
}

// NewSession creates a new execution session with the associated graph.
// options may be nil to use the default options.
func NewSession(graph *Graph, options *SessionOptions) (*Session, error) {
	status := newStatus()
	cOpt, doneOpt, err := options.c()
	defer doneOpt()
	if err != nil {
		return nil, err
	}
	cSess := C.TF_NewSession(graph.c, cOpt, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}

	s := &Session{c: cSess}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return s, nil
}

// Device structure contains information about a device associated with a session, as returned by ListDevices()
type Device struct {
	Name, Type       string
	MemoryLimitBytes int64
}

// String describes d and implements fmt.Stringer.
func (d Device) String() string {
	memStr := "no memory limit"
	if d.MemoryLimitBytes >= 0 {
		memStr = fmt.Sprintf("memory limit %d bytes", d.MemoryLimitBytes)
	}
	return fmt.Sprintf("(Device: name \"%s\", type %s, %s)", d.Name, d.Type, memStr)
}

func deviceSliceFromDeviceList(list *C.TF_DeviceList) ([]Device, error) {
	var devices []Device
	status := newStatus()

	for i := 0; i < int(C.TF_DeviceListCount(list)); i++ {
		name := C.TF_DeviceListName(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListName(index=%d) failed: %v", i, err)
		}

		deviceType := C.TF_DeviceListType(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListType(index=%d) failed: %v", i, err)
		}

		memoryLimitBytes := C.TF_DeviceListMemoryBytes(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListMemoryBytes(index=%d) failed: %v", i, err)
		}

		device := Device{
			Name:             C.GoString(name),
			Type:             C.GoString(deviceType),
			MemoryLimitBytes: int64(memoryLimitBytes),
		}

		devices = append(devices, device)
	}

	return devices, nil
}

// ListDevices returns the list of devices associated with a Session.
func (s *Session) ListDevices() ([]Device, error) {
	status := newStatus()
	devicesList := C.TF_SessionListDevices(s.c, status.c)
	if err := status.Err(); err != nil {
		return nil, fmt.Errorf("SessionListDevices() failed: %v", err)
	}
	defer C.TF_DeleteDeviceList(devicesList)
	return deviceSliceFromDeviceList(devicesList)
}

// Run the graph with the associated session starting with the supplied feeds
// to compute the value of the requested fetches. Runs, but does not return
// Tensors for operations specified in targets.
//
// On success, returns the fetched Tensors in the same order as supplied in
// the fetches argument. If fetches is set to nil, the returned Tensor fetches
// is empty.
func (s *Session) Run(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) ([]*Tensor, error) {
	s.mu.Lock()
	if s.c == nil {
		s.mu.Unlock()
		return nil, errors.New("session is closed")
	}
	s.wg.Add(1)
	s.mu.Unlock()
	defer s.wg.Done()

	c := newCRunArgs(feeds, fetches, targets)
	status := newStatus()
	C.TF_SessionRun(s.c, nil,
		ptrOutput(c.feeds), ptrTensor(c.feedTensors), C.int(len(feeds)),
		ptrOutput(c.fetches), ptrT