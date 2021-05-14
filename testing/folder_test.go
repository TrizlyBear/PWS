package testing

import (
	"fmt"
	dataset "github.com/TrizlyBear/PWS"
	"testing"
)

func TestFolder(t *testing.T) {
	set, err := dataset.FromFolder("../datasets/Training", dataset.Resize(100,100))
	if err != nil {
		panic(err)
	}

	fmt.Println(len(set.X))

}
