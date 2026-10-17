// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal bulk streaming kernel with @_force@ (generated from BulkStreamingMethodGPU.metal.inc)

#include "hoomd/mpcd/@_force@.h"
#include "hoomd/mpcd/BulkGeometry.h"

#define GEOMETRY_CLASS detail::BulkGeometry
#define FORCE_CLASS @_force@

#include "hoomd/mpcd/BounceBackStreamingMethodGPUMetal.h"
