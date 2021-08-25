package testing

import (
	"fmt"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"testing"
)

func TestFC(t *testing.T) {
	pred := [][][]float64{{{.5}},{{.3}},{{.2}},{{.1}}}
	fw := layers.FC{Out: 3}
	fmt.Println(fw.Forward(pred))

	fmt.Println(fw.Backward([][][]float64{
		{{.2}},{{.3}},{{.1}},
	},.1))
	//fmt.Println(math.Transpose([][]float64{{1},{2}}))
}
