/*
 * Convolutional Neural Network in C from scratch
 *
 * Copyright (C) 2026 Sanskar Rai 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include "common.h"
#include "error.h"

typedef struct {
    usize *shape;
    usize rank;

    usize *strides;

    usize size;
    float *data;
} Tensor;

Error tensor_create(Tensor *t, usize rank, const usize *shape);
void tensor_destroy(Tensor *t);

Error tensor_copy(Tensor *dest, const Tensor *src);
Error tensor_fill(Tensor *t, float val);

Error tensor_get(float *val, const Tensor *t, const usize *index);
Error tensor_set(Tensor *t, const usize *index, float val);

Error tensor_max(float *ret, const Tensor *t);
Error tensor_min(float *ret, const Tensor *t);
Error tensor_sum(float *ret, const Tensor *t);

Error tensor_hadamard_multiply(Tensor *result, const Tensor *a, const Tensor *b);
Error tensor_scalar_multiply(Tensor *result, const Tensor *t, float scalar);

Error tensor_randomize(Tensor *t, float min, float max);
Error tensor_he_uniform(Tensor *t, usize fan_in);
Error tensor_xavier_uniform(Tensor *t, usize fan_in, usize fan_out);

Error tensor_flatten(Tensor *t);
Error tensor_reshape(Tensor *t, usize rank, const usize *shape);

#endif /* TENSOR_H */
