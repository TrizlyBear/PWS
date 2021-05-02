package layers

import (
	"github.com/TrizlyBear/PWS/math"
)

type Flatten struct {
	y int
	x int
}

func (e *Flatten) Forward(in [][]float64) [][]float64 {
	e.y = len(in)
	e.x = len(in[0])
	out := []float64{}
	for _, x := range in {
		for _, y := range x {
			out = append(out, y)
		}

	}
	return [][]float64{out}
}

func (e *Flatten) Backward(err [][]float64, float642 float64) [][]float64 {
	return math.Resize(err, e.y, e.x)
}
