package activation

import (
	math2 "github.com/TrizlyBear/PWS/math"
	"math"
)

// Rectifier linear unit layer
type ReLu struct {
	input [][][]float64
}

// ReLu forward
func (e *ReLu) Forward(in [][][]float64) [][][]float64 {
	(*e).input = math2.Zeros(len(in),len(in[0]),len(in[0][0])).([][][]float64)

	for z,_ := range in {
		for y,_ := range in[z] {
			for x,_ := range in[z][y] {
				if in[z][y][x] >= 0 {
					(*e).input[z][y][x] = 1
				} else {
					in[z][y][x] = 0
				}
			}
		}
	}
	return in
}

// ReLu backward function
func (e *ReLu) Backward(err [][][]float64, lr float64) [][][]float64 {
	for z,_ := range err {
		for y,_ := range err[z] {
			for x,_ := range err[z][y] {
				if e.input[z][y][x] < 0 {
					err[z][y][x] = 0
				}
			}
		}
	}

	return err
}

// Hyperbolic tangent layer
type Tanh struct {
	Input [][][]float64
}

// Tanh Forward function
func (e *Tanh) Forward(in [][][]float64) [][][]float64 {
	(*e).Input = in

	for x,_ := range in {
		for y,_ := range in[x] {
			for z,_ := range in[x][y] {
				// Set every value to tanh(value)
				in[x][y][z] = math.Tanh(in[x][y][z])
			}
		}
	}

	return in
}

// Tanh backward function
func (e Tanh) Backward(err [][][]float64, lr float64) [][][]float64 {
	for x,_ := range err {
		for y,_ := range err[x] {
			for z,_ := range err[x][y] {
				// Multiply the error with the tanh derivative of itself
				err[x][y][z] *= (1 - math.Pow(math.Tanh(e.Input[x][y][z]),2))
			}
		}
	}

	return err
}

type Sigmoid struct {
	Input [][][]float64
}