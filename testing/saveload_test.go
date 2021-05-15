package testing

import (
	"fmt"
	dataset "github.com/TrizlyBear/PWS"
	"testing"
)

func TestSaveLoad(t *testing.T) {
	ds := &dataset.Dataset{}
	ds.X = [][][]float64{{{0.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}}
	ds.Y = [][][]float64{{{0.0}}, {{1.0}}, {{1.0}}, {{0.0}}}

	err := dataset.SaveDS(ds, "../datasets/test.bin")
	if err != nil {
		panic(err)
	}

	newds, err := dataset.LoadDS("../datasets/test.bin")
	fmt.Println(newds.Y)
}
