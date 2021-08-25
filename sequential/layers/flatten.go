package layers

import (
	"github.com/TrizlyBear/PWS/math"
)

// Flatten layer
type Flatten struct {
	depth 	int
	y 		int
	x 		int
}

// Flattern layer forward function
func (e *Flatten) Forward(in [][][]float64) [][][]float64 {

	//fmt.Println(in)
	// Saves the dimensions for the error computation
	e.depth = len(in)
	e.y = len(in[0])
	e.x = len(in[0][0])
	
	out := [][][]float64{}
	for _, x := range in {
		for _, y := range x {
			for _,z := range y {
				out = append(out, [][]float64{{z}})
			}

		}
	}
	return out
}

// Flatten backward function
func (e *Flatten) Backward(err [][][]float64, float642 float64) [][][]float64 {
	presort := make([][]float64, e.depth)

	for i := 0; i < e.depth; i++ {
		for o := 0; o < (len(err) / e.depth); o++ {
			presort[i] = append(presort[i], err[(i * e.depth) + o][0][0])
		}
	}

	out := [][][]float64{}

	for _,x := range presort {
		out = append(out, math.Resize([][]float64{x}, e.y, e.x))
	}

	// Resize the output to the input dimensions
	return out
}
