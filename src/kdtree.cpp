// Kdtree.cpp
#include "sim_local/kdtree.hpp"

// Constructor: split vector_db into map_desc_ and map_pts_, then build tree.
Kdtree::Kdtree(const cv::Mat& vector_db, int K)
  : K_(K)
{
    // matrix must be float and 2D
    assert(vector_db.type() == CV_32F);
    int C = vector_db.cols;
    assert(C == 183 || C == 184);

    int offset = (C == 184 ? 1 : 0);
    int descDim = 180;
    // allocate
    int N = vector_db.rows;
    map_desc_.reserve(N);
    map_pts_.reserve(N);

    for (int i = 0; i < N; ++i) {
        // extract one row
        const float* ptr = vector_db.ptr<float>(i);
        // descriptor
        std::vector<float> d(descDim);
        std::copy(ptr + offset, ptr + offset + descDim, d.begin());
        map_desc_.push_back(std::move(d));
        // position XYZ
        pcl::PointXYZ p;
        p.x = ptr[offset + descDim + 0];
        p.y = ptr[offset + descDim + 1];
        p.z = ptr[offset + descDim + 2];
        map_pts_.push_back(p);
    }

    // now build KD-tree on map_pts_
    build();
}

// Build KD-tree helper
void Kdtree::build() {
    if (map_pts_.empty()) return;
    // prepare indices
    std::vector<int> idxList(map_pts_.size());
    for (int i = 0; i < static_cast<int>(idxList.size()); ++i)
        idxList[i] = i;
    nodes_.clear();
    nodes_.reserve(map_pts_.size());
    root_ = buildRec(idxList, 0, idxList.size(), 0);
}

// Recursive construction
int Kdtree::buildRec(
    std::vector<int>& idxList,
    size_t l, size_t r,
    int depth)
{
    if (l >= r) return -1;
    int axis = depth % posDim_;
    size_t m = (l + r) / 2;
    std::nth_element(
        idxList.begin() + l,
        idxList.begin() + m,
        idxList.begin() + r,
        [&](int a, int b) {
            const auto& pa = map_pts_[a];
            const auto& pb = map_pts_[b];
            if (axis == 0) return pa.x < pb.x;
            if (axis == 1) return pa.y < pb.y;
            return pa.z < pb.z;
        });
    int nodeIdx = static_cast<int>(nodes_.size());
    nodes_.push_back({ idxList[m], -1, -1 });
    nodes_[nodeIdx].left  = buildRec(idxList, l,   m,   depth + 1);
    nodes_[nodeIdx].right = buildRec(idxList, m+1, r,   depth + 1);
    return nodeIdx;
}

// Query KNN: for each frame point, returns up to K_ map indices
std::vector<std::vector<int>> Kdtree::queryKNN(
    const std::vector<pcl::PointXYZ>& frame_pts) const
{
    size_t F = frame_pts.size();
    std::vector<std::vector<int>> result(F);
    for (size_t i = 0; i < F; ++i) {
        std::priority_queue<std::pair<float,int>> pq;
        float maxDist = 0.0f;
        searchKNN(frame_pts[i], root_, 0, pq, maxDist);
        result[i].reserve(pq.size());
        while (!pq.empty()) {
            result[i].push_back(pq.top().second);
            pq.pop();
        }
        std::reverse(result[i].begin(), result[i].end());
    }
    return result;
}

// Recursive KNN search
void Kdtree::searchKNN(
    const pcl::PointXYZ& query,
    int nodeIdx,
    int depth,
    std::priority_queue<std::pair<float,int>>& pq,
    float& maxDist) const
{
    if (nodeIdx < 0) return;
    const auto& nd = nodes_[nodeIdx];
    float dist = sqDist(query, map_pts_[nd.idx]);
    if ((int)pq.size() < K_) {
        pq.emplace(dist, nd.idx);
        if ((int)pq.size() == K_) maxDist = pq.top().first;
    } else if (dist < maxDist) {
        pq.pop();
        pq.emplace(dist, nd.idx);
        maxDist = pq.top().first;
    }
    int axis = depth % posDim_;
    float diff = (axis == 0 ? query.x - map_pts_[nd.idx].x
                : axis == 1 ? query.y - map_pts_[nd.idx].y
                            : query.z - map_pts_[nd.idx].z);
    int first  = (diff < 0 ? nd.left  : nd.right);
    int second = (diff < 0 ? nd.right : nd.left);
    if (first >= 0)  searchKNN(query, first,  depth+1, pq, maxDist);
    if (second>= 0 && ((int)pq.size()<K_ || diff*diff<maxDist))
                      searchKNN(query, second, depth+1, pq, maxDist);
}

// Squared Euclidean distance
float Kdtree::sqDist(
    const pcl::PointXYZ& a,
    const pcl::PointXYZ& b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}