#include "nnf.hpp"

class Shuffle {
public:
	Mat_<uint8_t> mask;
	Mat_<Vec3f> source;
	Mat_<Vec3f> target;
	Vec2i initalOffset;
	Vec2i offset;
	vector<Nnf> nnfs;
	Vec2i clickedPos;
	bool isMoving = false;
	const int pyramidLevels = 4;

	Shuffle(Mat_<uint8_t> m, Mat_<Vec3f> s, Vec2i o) {
		mask = m;
		source = s;
		target = s.clone();
		offset = o;
		initalOffset = o;
	}

	

	void shuffleInitalize(int constraintValue, bool swap);
	Mat_<Vec3b>  gaussianPyramids(int iterations, int alpha, int improvementIterations);
	static void doMouse(int event, int x, int y, int flags, void* userdata);
	void run();
};