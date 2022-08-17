#include "nnf.hpp"


void Nnf::nnfInitalize(int r, int c) {
	rows = r;
	cols = c;
	offsets = Mat_<Vec2i>(r, c);
	cost = Mat_<float>(r, c);
	for (int i = 0; i < rows; ++i) { // rows
		for (int j = 0; j < cols; ++j) { // cols
			int randR = rand() % rows;
			int randC = rand() % cols;
			offsets(i, j) = Vec2i(randR, randC);
			cost(i, j) = FLT_MAX;
		}
	}
}



Mat_<Vec3b> Nnf::reconstruct(const Mat_<Vec3f>& img) {
	Mat_<Vec3b> r = img.clone();
	for (int i = 0; i < rows; ++i) { // rows
		for (int j = 0; j < cols; ++j) { // cols
			Vec2i v = offsets(i, j);
			r(i, j) = img(v[0], v[1]);
		}
	}
	return r;
}


Mat_<Vec3b> Nnf::getOffsetMatrix() {
	return offsets;
}

Mat_<Vec3b> Nnf::nnf2image() {
	Mat_<Vec3b> m = Mat_<Vec3b>(rows, cols);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			Vec3b bgr = Vec3b{ 0,uint8_t((offsets(i,j)[0] * 255) / rows),uint8_t((offsets(i,j)[1] * 255) / cols) };
			bgr[0] = 255 - max(bgr[1], bgr[2]);
			m(i, j) = bgr;
		}
	return m;
}

inline float Nnf::ssd(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target, Vec2i sourcePos, Vec2i targetPos) {
	float dist = 0;
	if (sourcePos[0] >= HALF_PATCH && sourcePos[1] >= HALF_PATCH && targetPos[0] >= HALF_PATCH && targetPos[1] >= HALF_PATCH 
		&& sourcePos[0] < rows-HALF_PATCH && sourcePos[1] < cols-HALF_PATCH && targetPos[0] < rows - HALF_PATCH && targetPos[1] < cols - HALF_PATCH) {
		for (int i = -HALF_PATCH; i <= HALF_PATCH; i++) {
			for (int j = -HALF_PATCH; j <= HALF_PATCH; j++) {
				Vec3f pix1 = source((sourcePos[0] + i), (sourcePos[1] + j));
				Vec3f pix2 = target((targetPos[0] + i), (targetPos[1] + j));
				dist += diffsq(pix1, pix2);
			}
		}
		return dist;
	}
	for (int i = -HALF_PATCH; i <= HALF_PATCH; i++) {
		for (int j = -HALF_PATCH; j <= HALF_PATCH; j++) {

			Vec3f pix1 = source(clamp(sourcePos[0] + i,0,rows-1),clamp(sourcePos[1] + j,0,cols-1));
			Vec3f pix2 = target(clamp(targetPos[0] + i, 0, rows - 1), clamp(targetPos[1] + j, 0, cols - 1));
			dist += diffsq(pix1,pix2);
		}
	}
	return dist;
}

void Nnf::improveNnf(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target, Vec2i sourcePos, Vec2i targetPos) {
	if (cost(targetPos) == 0) {
		return;
	}
	if (offsets(targetPos) == sourcePos) {
		return;
	}
	float f = ssd(source,target,sourcePos,targetPos);
	if (f < cost(targetPos)) {
		cost(targetPos) = f;
		offsets(targetPos) = sourcePos;
	}
}

void Nnf::randomSearch(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target,int alpha) {
	initalizeCost(source,target);

	int radius;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			radius = max(cols, rows);
			while (radius > 0) {

				Vec2i off = offsets(i, j);
				int rMin = off[0] - radius;
				int rMax = off[0] + radius + 1;
				int cMin = off[1] - radius;
				int cMax = off[1] + radius + 1;
				rMin = max(rMin, 0);
				cMin = max(cMin, 0);
				rMax = min(rMax, rows - 1);
				cMax = min(cMax, cols - 1);
				int rr = rMin + rand() % (rMax - rMin);
				int rc = cMin + rand() % (cMax - cMin);
				improveNnf(source, target, Vec2i(rr, rc), Vec2i(i,j));
				radius = radius / alpha;
			}
		}
	}

}

void Nnf::randomSearchIndividual(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target, int alpha,int i, int j) {
	int radius;
			radius = max(cols, rows);
			while (radius > 0) {

				Vec2i off = offsets(i, j);
				int rMin = off[0] - radius;
				int rMax = off[0] + radius + 1;
				int cMin = off[1] - radius;
				int cMax = off[1] + radius + 1;
				rMin = max(rMin, 0);
				cMin = max(cMin, 0);
				rMax = min(rMax, rows - 1);
				cMax = min(cMax, cols - 1);
				int rr = rMin + rand() % (rMax - rMin);
				int rc = cMin + rand() % (cMax - cMin);
				improveNnf(source, target, Vec2i(rr, rc), Vec2i(i, j));
				radius = radius / alpha;
			}
}

void Nnf::propagate(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target, int iterations, int alpha) {
	initalizeCost(source,target);
	int previousCol;
	int previousRow;
	Vec2i previousColOffset;
	Vec2i previousRowOffset;
	int direction = 1;
	for (int count = 0; count < iterations; count++) {
		for (int i = 0; i != rows; i++) {
			for (int j = 0; j != cols; j++) {
				//Columns
				previousCol = j - direction;
				previousCol = clamp(previousCol, 0, cols - 1);
				previousColOffset = offsets(i, previousCol);
				int adjustedColOffset = previousColOffset[1] + direction;
				adjustedColOffset = clamp(adjustedColOffset, 0, cols - 1);
				improveNnf(source, target, Vec2i(previousColOffset[0], adjustedColOffset), Vec2i(i, j));
				//Rows
				previousRow = i - direction;
				previousRow = clamp(previousRow, 0, rows - 1);
				previousRowOffset = offsets(previousRow, j);
				int adjustedRowOffset = previousRowOffset[0] + direction;
				adjustedRowOffset = clamp(adjustedRowOffset, 0, rows - 1);
				improveNnf(source, target, Vec2i(adjustedRowOffset, previousRowOffset[1]), Vec2i(i, j));
				randomSearchIndividual(source,target,alpha,i,j);
			}
		}
		direction = -1;
		for (int i = rows - 1; i != -1; i--) {
			for (int j = cols - 1; j != -1; j--) {
				//Columns
				previousCol = j - direction;
				previousCol = clamp(previousCol, 0, cols - 1);
				previousColOffset = offsets(i, previousCol);
				int adjustedColOffset = previousColOffset[1] + direction;
				adjustedColOffset = clamp(adjustedColOffset, 0, cols - 1);
				improveNnf(source, target, Vec2i(previousColOffset[0], adjustedColOffset), Vec2i(i, j));
				//Rows
				previousRow = i - direction;
				previousRow = clamp(previousRow, 0, rows - 1);
				previousRowOffset = offsets(previousRow, j);
				int adjustedRowOffset = previousRowOffset[0] + direction;
				adjustedRowOffset = clamp(adjustedRowOffset, 0, rows - 1);
				improveNnf(source, target, Vec2i(adjustedRowOffset, previousRowOffset[1]), Vec2i(i, j));
				randomSearchIndividual(source, target, alpha, i, j);
			}
		}
	}
}

void Nnf::initalizeCost(const Mat_<Vec3f>& source, const Mat_<Vec3f>& target) {
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			Vec2i off = offsets(i, j);
			cost(i, j) = ssd(source, target, off,Vec2i( i, j));
		}
}

Mat_<Vec3f> Nnf::patchReconstruction(const Mat_<Vec3f>& source) {
	Mat_<Vec3f> reconstruction = Mat_<Vec3f>::zeros(rows, cols);
	Mat_<int> count = Mat_<int>::zeros(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Vec2i off = offsets(i, j);
			for (int k = -HALF_PATCH; k <= HALF_PATCH;k++) {
				int ci = clamp(off[0] + k, 0, rows - 1);
				for (int l = -HALF_PATCH; l < +HALF_PATCH;l++) {
					if (inBounds(i + k, j + l)) {
					int cj = clamp(off[1] + l, 0, cols - 1);
					reconstruction(i + k, j + l) += source(ci, cj);
					count(i + k, j + l)++;
					}
				}
			}
		}
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			reconstruction(i, j) /= count(i, j);
		}
	}
	return reconstruction;
}

Nnf Nnf::upSample(Nnf up) {
	cout << "upsample" << float(up.rows) / rows << "," << float(up.cols) / cols << endl;
	Mat_<Vec2f> foff(offsets);
	for (int i = 0; i < up.rows; i++)
		for (int j = 0; j < up.cols; j++) {
			int xo = offsets(i/2, j/2)[1] * 2 + j%2;
			int yo = offsets(i/2, j/2)[0] * 2 + i%2;
			xo = min(max(xo, 0), up.cols - 1);
			yo = min(max(yo, 0), up.rows - 1);
			up.offsets(i, j) = { yo,xo };
		}
	return up;
}
