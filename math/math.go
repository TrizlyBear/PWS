package math

import (
	"math"
	"sort"
)

// Calculates the mean value of the input array
func Mean(in []float64) float64 {
	total := 0.0
	for _, v := range in {
		total += v
	}
	return total / float64(len(in))
}

// Returns the biggest number in the array
func Max(in []float64) float64 {
	sort.Float64s(in)
	return in[len(in)-1]
}

func MaxIndex(in [][]float64) (float64, int, int) {
	biggest := math.Inf(-1)
	bY := -1
	bX := -1
	for y,_ := range in {
		for x,_ := range in[y] {
			if in[y][x] > biggest {
				biggest = in[y][x]
				bY = y
				bX = x
			}
		}
	}

	return biggest, bX, bY
}

// Calculates the sum of the input array
func Sum(in []float64) float64 {
	sum := .0
	for _, x := range in {
		sum += x
	}
	return sum
}
