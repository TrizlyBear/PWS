package testing

import (
	"fmt"
	dataset2 "github.com/TrizlyBear/PWS/utils/dataset"
	"testing"
)

func TestDataset(t *testing.T) {
	ds, err := dataset2.FromCSV("../datasets/mnist_train.csv",[]int{0}, dataset2.Max(2))
	if err != nil {
		panic(err)
	}
	ds.Reshape(28,28)
	fmt.Println(ds.Split(1))
}
