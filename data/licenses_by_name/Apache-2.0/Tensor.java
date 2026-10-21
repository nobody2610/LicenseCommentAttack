/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow;

import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.HashMap;

/**
 * A statically typed multi-dimensional array whose elements are of a type described by T.
 *
 * <p>Instances of a Tensor are <b>not</b> thread-safe.
 *
 * <p><b>WARNING:</b> Resources consumed by the Tensor object <b>must</b> be explicitly freed by
 * invoking the {@link #close()} method when the object is no longer needed. For example, using a
 * try-with-resources block:
 *
 * <pre>{@code
 * try (Tensor t = Tensor.create(...)) {
 *   doSomethingWith(t);
 * }
 * }</pre>
 */
public final class Tensor<T> implements AutoCloseable {

  /**
   * Creates a Tensor from a Java object.
   *
   * <p>A {@code Tensor} is a multi-dimensional array of elements of a limited set of types. Not all
   * Java objects can be converted to a {@code Tensor}. In particular, the argument {@code obj} must
   * be either a primitive (float, double, int, long, boolean, byte) or a multi-dimensional array of
   * one of those primitives. The argument {@code type} specifies how to interpret the first
   * argument as a TensorFlow type. For example:
   *
   * <pre>{@code
   * // Valid: A 64-bit integer scalar.
   * Tensor<Long> s = Tensor.create(42L, Long.class);
   *
   * // Valid: A 3x2 matrix of floats.
   * float[][] matrix = new float[3][2];
   * Tensor<Float> m = Tensor.create(matrix, Float.class);
   *
   * // Invalid: Will throw an IllegalArgumentException as an arbitrary Object
   * // does not fit into the TensorFlow type system.
   * Tensor<?> o = Tensor.create(new Object())
   *
   * // Invalid: Will throw an IllegalArgumentException since there are
   * // a differing number of elements in each row of this 2-D array.
   * int[][] twoD = new int[2][];
   * twoD[0] = new int[1];
   * twoD[1] = new int[2];
   * Tensor<Integer> x = Tensor.create(twoD, Integer.class);
   * }</pre>
   *
   * {@link String}-typed Tensors are multi-dimensional arrays of arbitrary byte sequences, so can
   * be initialized from arrays of {@code byte[]} elements. For example:
   *
   * <pre>{@code
   * // Valid: A String tensor.
   * Tensor<String> s = Tensor.create(new byte[]{1, 2, 3}, String.class);
   *
   * // Java Strings will need to be encoded into a byte-sequence.
   * String mystring = "foo";
   * Tensor<String> s = Tensor.create(mystring.getBytes("UTF-8"), String.class);
   *
   * // Valid: Matrix of String tensors.
   * // Each element might have a different length.
   * byte[][][] matrix = new byte[2][2][];
   * matrix[0][0] = "this".getBytes("UTF-8");
   * matrix[0][1] = "is".getBytes("UTF-8");
   * matrix[1][0] = "a".getBytes("UTF-8");
   * matrix[1][1] = "matrix".getBytes("UTF-8");
   * Tensor<String> m = Tensor.create(matrix, String.class);
   * }</pre>
   *
   * @param obj The object to convert to a {@code Tensor<T>}. Note that whether it is compatible
   *     with the type T is not checked by the type system. For type-safe creation of tensors, use
   *     {@link Tensors}.
   * @param type The class object representing the type T.
   * @throws IllegalArgumentException if {@code obj} is not compatible with the TensorFlow type
   *     system.
   */
  @SuppressWarnings("unchecked")
  public static <T> Tensor<T> create(Object obj, Class<T> type) {
    DataType dtype = DataType.fromClass(type);
    if (!objectCompatWithType(obj, dtype)) {
      throw new IllegalArgumentException(
          "DataType of object does not match T (expected "
              + dtype
              + ", got "
              + dataTypeOf(obj)
              + ")");
    }
    return (Tensor<T>) create(obj, dtype);
  }

  /**
   * Creates a tensor from an object whose class is inspected to figure out what the underlying data
   * type should be.