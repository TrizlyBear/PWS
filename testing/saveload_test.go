package testing

import (
	"fmt"
	dataset2 "github.com/TrizlyBear/PWS/utils/dataset"
	"testing"
)

func TestSaveLoad(t *testing.T) {
	ds := &dataset2.Dataset{}
	ds.X = [][][]float64{{{0.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}}
	ds.Y = [][][]float64{{{0.0}}, {{1.0}}, {{1.0}}, {{0.0}}}

	err := dataset2.SaveDS(ds, "../datasets/test.bin")
	if err != nil {
		panic(err)
	}

	newds, err := dataset2.LoadDS("../datasets/test.bin")
	fmt.Println(newds.Y)
}
