package cnn

import (
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
