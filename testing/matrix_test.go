package testing

import (
	"fmt"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"testing"
)

func TestRotate180(t *testing.T) {
	in := [][]float64{
		{1,2},
		{3,4},
	}

	l := layers.TestConv{
		KernelSize: struct {
			X int
			Y int
		}{2,2},
		Depth:  2,
		Stride: 1,
	}
	fmt.Println(l.Forward([][][]float64{in}))
}
