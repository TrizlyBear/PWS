package cnn

import (
	"fmt"
	"sort"
)

func Mean(in []float64) float64 {
	total := 0.0
	fmt.Printf(("new"))
	for _, v := range in {
		fmt.Printf("%v", v)
		total += v
	}
	return total / float64(len(in))
}

func Max(in []float64) float64 {
	sort.Float64s(in)
	return in[len(in)-1]
}
