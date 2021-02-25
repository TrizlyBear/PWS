package math

import (
	"errors"
	"math/rand"
)

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

	out := make([][]float64, len(x))
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

func Reverse(s []interface{}) []interface{} {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
	return s
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

func Resize(matrix [][]float64, y int, x int) [][]float64 {
	out := [][]float64{}
	flat := []float64{}
	for _, i := range matrix {
		for _, o := range i {
			flat = append(flat, o)
		}

	}
	for i := 0; i < y; i++ {
		for o := 0; o < x; o++ {
			out[i][o] = flat[0]
			flat = flat[1:]
		}
	}
	return out
}
