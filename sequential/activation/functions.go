package activation

import (
	"math"
)

// Rectifier linear unit layer
type ReLu struct {
	input [][]float64
}

// ReLu forward
func (e *ReLu) Forward(in [][]float64) [][]float64 {
	(*e).input = in
	for y,el := range in {
		for x,_ := range el {
			
			in[y][x] = math.Max(0, in[y][x])
		}
	}
	return in
}

// ReLu backward function
func (e *ReLu) Backward(err [][]float64, lr float64) [][]float64 {
	for y,el := range err {
		for x,_ := range el {
			err[y][x] = (1 - math.Pow(math.Tanh(err[y][x]),2)) * err[y][x]
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
