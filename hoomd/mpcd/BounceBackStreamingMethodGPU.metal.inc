// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal confined streaming kernel with @_geometry@ and @_force@ (generated from
// BounceBackStreamingMethodGPU.metal.inc)

#include "hoomd/mpcd/@_force@.h"
#include "hoomd/mpcd/@_geometry@.h"

#define GEOMETRY_CLASS @_geometry@
#define FORCE_CLASS @_force@

#include "hoomd/mpcd/BounceBackStreamingMethodGPUMetal.h"
