package math

import (
	"reflect"
	"testing"
)

func TestRotate180(t *testing.T) {
	type args struct {
		in [][]float64
	}
	tests := []struct {
		name string
		args args
		want [][]float64
	}{
		{
			name: "Test1",
			args:args{in: [][]float64{{1,2},{3,4}}},
			want: [][]float64{{4,3},{2,1}},
		},
	}
		for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Rotate180(tt.args.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Rotate180() = %v, want %v", got, tt.want)
			}
		})
	}
}
