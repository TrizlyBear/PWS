package activation

import "math"

type ReLu struct {
	input [][]float64
}

func (e *ReLu) Forward(in [][]float64) [][]float64 {
	(*e).input = in
	for y,el := range in {
		for x,_ := range el {
			in[y][x] = math.Max(0, in[y][x])
		}
	}
	return in
}

func (e *ReLu) Backward(err [][]float64, lr float64) [][]float64 {
	for y,el := range err {
		for x,_ := range el {
			//if (*e).in {
			//
			//}
			err[y][x] = (1 - math.Pow(math.Tanh(err[y][x]),2)) * err[y][x]
		}
	}
	return err
}

type Tanh struct {
	Input [][]float64
}

func (e *Tanh) Forward(in [][]float64) [][]float64 {
	(*e).Input = in
	for y,el := range in {
		for x,_ := range el {
			in[y][x] = math.Tanh(in[y][x])
		}
	}
	return in
}

func (e Tanh) Backward(err [][]float64, lr float64) [][]float64 {
	for y,el := range err {
		for x,_ := range el {
			err[y][x] *= (1 - math.Pow(math.Tanh(e.Input[y][x]),2))
		}
	}
	return err
}
