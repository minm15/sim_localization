// Kdtree.hpp
#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <vector>
#include <queue>
#include <cassert>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>

/// KD-tree for 3D position pruning of descriptor matching.
/// Splits a descriptor matrix (cv::Mat) into descriptors and XYZ positions.
class Kdtree {
public:
    Kdtree(const cv::Mat& vector_db, int K = 10);

    /// Query K nearest neighbors for each point in frame_pts.
    /// Returns an [F x <=K] list of map indices.
    std::vector<std::vector<int>> queryKNN(
        const std::vector<pcl::PointXYZ>& frame_pts
    ) const;

private:
    struct Node { int idx, left, right; };

    int K_;                                  ///< number of neighbors
    int posDim_ = 3;                         ///< always 3 for XYZ
    int root_ = -1;                          ///< root node index
    std::vector<Node> nodes_;               ///< flat tree storage
    std::vector<pcl::PointXYZ> map_pts_;     ///< map positions
    std::vector<std::vector<float>> map_desc_; ///< map descriptors (180-d)

    /// Build KD-tree over map_pts_
    void build();

    /// Recursively build subtree [l,r)
    int buildRec(std::vector<int>& idxList, size_t l, size_t r, int depth);

    /// KNN search helper
    void searchKNN(const pcl::PointXYZ& query,
                   int nodeIdx,
                   int depth,
                   std::priority_queue<std::pair<float,int>>& pq,
                   float& maxDist) const;

    /// Squared Euclidean between 3D points
    static float sqDist(const pcl::PointXYZ& a, const pcl::PointXYZ& b);
};

#endif // KDTREE_HPP