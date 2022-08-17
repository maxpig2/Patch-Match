#pragma once

// std
#include <iostream>
#include <math.h>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct { int row; int col; } Pos;

using namespace cv;
using namespace std;


inline float diffsq(Vec3f p1, Vec3f p2) {
	return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]);
}




const int PATCH_SIZE = 7;
const int HALF_PATCH = PATCH_SIZE / 2;

class Nnf {
public:
	int rows;
	int	cols;


	Nnf() {}

	Nnf(int r, int c) {
		nnfInitalize(r,c);
	}

	Mat_<Vec2i> offsets;
	Mat_<float> cost;

	inline bool inBounds(int r, int c) {
		if (r > rows-1 || r < 0 || c < 0 || c > cols-1) {
			return false;
		}
		return true;
	}
	void nnfInitalize(int r, int c);

	Mat_<Vec3b> reconstruct(const Mat_<Vec3f>& img);
	Mat_<Vec3b> nnf2image();
	Mat_<Vec3b> getOffsetMatrix();
	Mat_<Vec3b> quilt(const Mat_<Vec3f>& source);
	float ssd(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target,Vec2i sourcePos, Vec2i targetPos);
	void improveNnf(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target, Vec2i sourcePos, Vec2i targetPos);
	void randomSearch(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target,int alpha);
	void randomSearchIndividual(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target, int alpha, int i, int j);
	void propagate(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target,int iterations,int alpha);
	void initalizeCost(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target);
	Mat_<Vec3f> Nnf::patchReconstruction(const Mat_<Vec3f>& source);
	Nnf upSample(Nnf up);
};
