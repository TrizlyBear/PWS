package cnn

import (
	"math"
)

type Flatten struct {
}

type ReLu struct {
}

func (e Flatten) Forward(in [][]float64) ([]float64, error) {
	out := []float64{}
	for _, x := range in {
		for _, y := range x {
			out = append(out, y)
		}

	}
	return out, nil
}

func (e ReLu) Forward(in [][]float64) ([][]float64, error) {
	out := in
	for x, l := range in {
		for y, i := range l {
			out[x][y] = math.Max(0, i)
		}
	}
	return out, nil
}
