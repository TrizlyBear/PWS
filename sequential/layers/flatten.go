package layers

import (
	"github.com/TrizlyBear/PWS/math"
)

// Flatten layer
type Flatten struct {
	y int
	x int
}

// Flattern layer forward function
func (e *Flatten) Forward(in [][]float64) [][]float64 {
	// Saves the dimensions for the error computation
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

// Flatten backward function
func (e *Flatten) Backward(err [][]float64, float642 float64) [][]float64 {
	// Resize the output to the input dimensions
	return math.Resize(err, e.y, e.x)
}
