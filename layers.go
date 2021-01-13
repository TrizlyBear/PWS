package cnn

import (
	"fmt"
	"math"
)

type Flatten struct {
}

type ReLu struct {
}

type FC struct {
	Biases  [][]float64 `json: "omitempty"`
	Weights [][]float64 `json: "omitempty"`
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

func (e FC) Forward(in [][]float64, out int, insize int) ([][]float64, error) {
	if len(e.Biases) == 0 {
		fmt.Print("No biases")
		for i := 0; i < out; i++ {
			e.Biases = Rand(1, 1)
			e.Weights = Rand(insize, out)
		}
	}
	output, _ := Dot(in, e.Weights)
	for y, el := range output {
		for x, _ := range el {
			output[y][x] += e.Biases[0][0]
		}
	}

	return output, nil
}

func (e FC) Backward(error float64) {

}
