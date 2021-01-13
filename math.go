package cnn

import (
	"errors"
	"math/rand"
	"sort"
)

func Mean(in []float64) float64 {
	total := 0.0
	for _, v := range in {
		total += v
	}
	return total / float64(len(in))
}

func Max(in []float64) float64 {
	sort.Float64s(in)
	return in[len(in)-1]
}

func Sum(in []float64) float64 {
	sum := .0
	for _, x := range in {
		sum += x
	}
	return sum
}

func Rand(x int, y int) [][]float64 {

	out := make([][]float64, int(y))
	for i := 0; i < x; i++ {
		for m := 0; m < y; m++ {
			out[m] = append(out[m], rand.Float64()-0.5)
		}
	}
	return out
}

func Dot(x [][]float64, y [][]float64) (r [][]float64, err error) {
	if len(x[0]) != len(y) {
		return nil, errors.New("Dimensions not valid")
	}
	out := make([][]float64, len(x), len(y[0]))
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(y[0]); j++ {
			out[i] = append(out[i], 0)
			for k := 0; k < len(x[0]); k++ {
				out[i][j] += (x[i][k] * y[k][j])
			}
		}
	}

	return out, nil
}

func Transpose(slice [][]float64) [][]float64 {
	xl := len(slice[0])
	yl := len(slice)
	result := make([][]float64, xl)
	for i := range result {
		result[i] = make([]float64, yl)
	}
	for i := 0; i < xl; i++ {
		for j := 0; j < yl; j++ {
			result[i][j] = slice[j][i]
		}
	}
	return result
}
