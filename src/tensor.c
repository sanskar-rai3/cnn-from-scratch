/*
 * Neural Network From Scratch in C
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

#include "tensor.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static bool tensor_is_valid(const Tensor *t) {
    return t && t->rank > 0 && t->shape && t->strides && t->data && t->size > 0;
}

static bool tensor_size(usize rank, const usize *shape, usize *size) {
    if (!rank || !shape || !size)
        return false;

    usize total = 1;
    for (usize i = 0; i < rank; i++) {
        if (!shape[i] || total > SIZE_MAX / shape[i])
            return false;

        total *= shape[i];
    }

    if (total > SIZE_MAX / sizeof(float))
        return false;

    *size = total;

    return true;
}

static Error tensor_build_strides(usize *strides, usize rank, const usize *shape) {
    if (!strides || !rank || !shape)
        return ERROR_INVALID_ARG;

    strides[rank - 1] = 1;
    for (usize i = rank - 1; i > 0; i--) {
        if (strides[i] > SIZE_MAX / shape[i])
            return ERROR_INVALID_ARG;

        strides[i - 1] = strides[i] * shape[i];
    }

    return ERROR_OK;
}

Error tensor_create(Tensor *t, usize rank, const usize *shape) {
    if (!t || !shape || !rank)
        return ERROR_INVALID_ARG;

    usize size;
    if (!tensor_size(rank, shape, &size))
        return ERROR_INVALID_ARG;

    Tensor tmp = {0};

    if (rank > SIZE_MAX / sizeof(usize))
        return ERROR_INVALID_ARG;

    tmp.shape = malloc(rank * sizeof(*tmp.shape));
    if (!tmp.shape)
        return ERROR_ALLOCATION_FAILED;

    memcpy(tmp.shape, shape, rank * sizeof(*tmp.shape));

    tmp.strides = malloc(rank * sizeof(*tmp.strides));
    if (!tmp.strides) {
        free(tmp.shape);

        return ERROR_ALLOCATION_FAILED;
    }

    Error err = tensor_build_strides(tmp.strides, rank, tmp.shape);
    if (err != ERROR_OK) {
        free(tmp.strides);
        free(tmp.shape);
        return err;
    }

    tmp.data = malloc(size * sizeof(*tmp.data));
    if (!tmp.data) {
        free(tmp.strides);
        free(tmp.shape);

        return ERROR_ALLOCATION_FAILED;
    }

    tmp.rank = rank;
    tmp.size = size;

    *t = tmp;

    return ERROR_OK;
}

void tensor_destroy(Tensor *t) {
    if (!t)
        return;

    free(t->data);
    free(t->shape);
    free(t->strides);

    *t = (Tensor){0};
}

Error tensor_copy(Tensor *dest, const Tensor *src) {
    if (!dest || !tensor_is_valid(src) || dest == src)
        return ERROR_INVALID_ARG;

    Tensor tmp = {0};
    Error err = tensor_create(&tmp, src->rank, src->shape);
    if (err != ERROR_OK)
        return err;

    memcpy(tmp.data, src->data, src->size * sizeof(*src->data));

    tensor_destroy(dest);
    *dest = tmp;

    return ERROR_OK;
}

Error tensor_fill(Tensor *t, float val) {
    if (!tensor_is_valid(t))
        return ERROR_INVALID_ARG;

    for (usize i = 0; i < t->size; i++)
        t->data[i] = val;

    return ERROR_OK;
}

Error tensor_get(float *val, const Tensor *t, const usize *index) {
    if (!val || !tensor_is_valid(t) || !index)
        return ERROR_INVALID_ARG;

    usize offset = 0;
    for (usize i = 0; i < t->rank; i++) {
        if (index[i] >= t->shape[i])
            return ERROR_INVALID_ARG;

        offset += index[i] * t->strides[i];
    }

    *val = t->data[offset];
    return ERROR_OK;
}

Error tensor_set(Tensor *t, const usize *index, float val) {
    if (!tensor_is_valid(t) || !index)
        return ERROR_INVALID_ARG;

    usize offset = 0;
    for (usize i = 0; i < t->rank; i++) {
        if (index[i] >= t->shape[i])
            return ERROR_INVALID_ARG;

        offset += index[i] * t->strides[i];
    }

    t->data[offset] = val;
    return ERROR_OK;
}

Error tensor_max(float *ret, const Tensor *t) {
    if (!ret || !tensor_is_valid(t))
        return ERROR_INVALID_ARG;

    float max = t->data[0];
    for (usize i = 1; i < t->size; i++) {
        if (t->data[i] > max)
            max = t->data[i];
    }

    *ret = max;
    return ERROR_OK;
}

Error tensor_min(float *ret, const Tensor *t) {
    if (!ret || !tensor_is_valid(t))
        return ERROR_INVALID_ARG;

    float min = t->data[0];
    for (usize i = 1; i < t->size; i++) {
        if (t->data[i] < min)
            min = t->data[i];
    }

    *ret = min;
    return ERROR_OK;
}

Error tensor_sum(float *ret, const Tensor *t) {
    if (!ret || !tensor_is_valid(t))
        return ERROR_INVALID_ARG;

    float sum = 0.0f;
    for (usize i = 0; i < t->size; i++)
        sum += t->data[i];

    *ret = sum;
    return ERROR_OK;
}

static bool tensor_same_shape(const Tensor *a, const Tensor *b) {
    if (!tensor_is_valid(a) || !tensor_is_valid(b) ||
        a->rank != b->rank || a->size != b->size)
        return false;

    for (usize i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i])
            return false;
    }

    return true;
}

Error tensor_hadamard_multiply(Tensor *result, const Tensor *a, const Tensor *b) {
    if (!tensor_same_shape(a, b) || !tensor_same_shape(result, a))
        return ERROR_INVALID_ARG;

    for (usize i = 0; i < result->size; i++)
        result->data[i] = a->data[i] * b->data[i];

    return ERROR_OK;
}

Error tensor_scalar_multiply(Tensor *result, const Tensor *t, float scalar) {
    if (!tensor_same_shape(result, t))
        return ERROR_INVALID_ARG;

    for (usize i = 0; i < result->size; i++)
        result->data[i] = scalar * t->data[i];

    return ERROR_OK;
}

static float random_float(float min, float max) {
    static bool seeded;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = true;
    }

    float scale = (float)rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

Error tensor_randomize(Tensor *t, float min, float max) {
    if (!tensor_is_valid(t) || !isfinite(min) || !isfinite(max) || min > max)
        return ERROR_INVALID_ARG;

    for (usize i = 0; i < t->size; i++)
        t->data[i] = random_float(min, max);

    return ERROR_OK;
}

Error tensor_he_uniform(Tensor *t, usize fan_in) {
    if (!tensor_is_valid(t) || !fan_in)
        return ERROR_INVALID_ARG;

    float limit = sqrtf(6.0f / (float)fan_in);
    return tensor_randomize(t, -limit, limit);
}

Error tensor_xavier_uniform(Tensor *t, usize fan_in, usize fan_out) {
    if (!tensor_is_valid(t) || !fan_in || !fan_out ||
        fan_in > SIZE_MAX - fan_out)
        return ERROR_INVALID_ARG;

    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    return tensor_randomize(t, -limit, limit);
}

Error tensor_flatten(Tensor *t) {
    if (!tensor_is_valid(t))
        return ERROR_INVALID_ARG;

    usize *shape = malloc(sizeof(*shape));
    if (!shape)
        return ERROR_ALLOCATION_FAILED;

    usize *strides = malloc(sizeof(*strides));
    if (!strides) {
        free(shape);
        return ERROR_ALLOCATION_FAILED;
    }

    shape[0] = t->size;
    strides[0] = 1;

    free(t->shape);
    free(t->strides);

    t->shape = shape;
    t->strides = strides;
    t->rank = 1;

    return ERROR_OK;
}

Error tensor_reshape(Tensor *t, usize rank, const usize *shape) {
    if (!tensor_is_valid(t) || !shape || !rank ||
        rank > SIZE_MAX / sizeof(usize))
        return ERROR_INVALID_ARG;

    usize size;
    if (!tensor_size(rank, shape, &size) || size != t->size)
        return ERROR_INVALID_ARG;

    usize *new_shape = malloc(rank * sizeof(*new_shape));
    if (!new_shape)
        return ERROR_ALLOCATION_FAILED;

    memcpy(new_shape, shape, rank * sizeof(*new_shape));

    usize *new_strides = malloc(rank * sizeof(*new_strides));
    if (!new_strides) {
        free(new_shape);
        return ERROR_ALLOCATION_FAILED;
    }

    Error err = tensor_build_strides(new_strides, rank, new_shape);
    if (err != ERROR_OK) {
        free(new_strides);
        free(new_shape);
        return err;
    }

    free(t->shape);
    free(t->strides);

    t->shape = new_shape;
    t->strides = new_strides;
    t->rank = rank;

    return ERROR_OK;
}