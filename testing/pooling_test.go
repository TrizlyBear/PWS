package testing

import (
	"fmt"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"testing"
)

func  TestPooling(t *testing.T)  {
	l := &activation.MaxPooling{}
	l.Forward([][][]float64{{
		{.1,.0,.0,.2},
		{.0,.0,.0,.0},
		{.0,.0,.0,.0},
		{.1,.0,.0,.2},
	},{
		{.1,.0,.0,.2},
		{.0,.7,.7,.0},
		{.0,.7,.7,.0},
		{.1,.0,.0,.2},
	}})
	fmt.Println(l.Backward([][][]float64{
		{{.1,.2},{.3,.4}},
		{{.1,.2},{.3,.4}},
	},.1))
}
