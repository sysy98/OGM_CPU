// Include guard
#ifndef sensor_processing_H
#define sensor_processing_H

// Includes
#include <ros/ros.h>
#include <Eigen/Dense>
#include <sensor_msgs/PointCloud2.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <message_filters/subscriber.h>

// Types of point and cloud to work with
typedef pcl::PointXYZ VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;

// Namespaces
namespace sensor_processing{

using namespace std;
using namespace Eigen;
using namespace sensor_msgs;
using namespace tf2_msgs;
using namespace geometry_msgs;
using namespace nav_msgs;
using namespace message_filters;

// Parameter handler
struct Parameters{

	std::string home_dir;
	std::string scenario;

	float grid_range_min;
	float grid_range_max;
	float grid_cell_size;
	int width_grid;
	int height_grid;
	int grid_segments;
	int grid_bins;

	int occ_width_grid;
    int occ_height_grid;

	float grid_cell_height;

	float inv_angular_res;
	float inv_radial_res;

	float lidar_height;
	float lidar_opening_angle;
	float lidar_z_min;

	float ransac_tolerance;
	int ransac_iterations;

};

// Attributes of cell from polar grid
struct PolarCell{

	float x_min, y_min, z_min;
	float z_max;
	float ground;
	float height;
	float dist;
	int count;
	int elevated_count;
	float p_occ;
	float p_free;
	float p_final;
	float p_logit;

	// Default constructor.
	PolarCell():
		z_min(0.0), z_max(0.0), height(0.0), dist(0.0), count(0), 
		elevated_count(0), p_occ(0.0), p_free(0.5), p_final(0.0), 
		p_logit(0.0)
	{}
};

class SensorFusion{

public:

	// Default constructor
	SensorFusion(ros::NodeHandle nh, ros::NodeHandle private_nh);

	// Virtual destructor
	virtual ~SensorFusion();

	// Processes 3D Velodyne point cloud and publishes the output grid message
	virtual void process(
		const PointCloud2::ConstPtr & cloud);

private:

	// Node handle
	ros::NodeHandle nh_, private_nh_;

	// Class members
	Parameters params_;
	float width_gain_;  // 2
    float height_gain_; // 3.5

	static int time_frame_;
    static ros::Time cloud_stamp_;
	static Eigen::Matrix4f transMat;

	VPointCloud::Ptr pcl_in_;
	VPointCloud::Ptr pcl_ground_plane_;
	VPointCloud::Ptr pcl_ground_plane_inliers_;
	VPointCloud::Ptr pcl_ground_plane_outliers_;
	VPointCloud::Ptr pcl_ground_;
	VPointCloud::Ptr pcl_elevated_;

	std::vector< std::vector<int> > polar_measurements_;
	std::vector< std::vector<PolarCell> > polar_grid_;
	std::vector< float > whole_grid_probs_;
	OccupancyGrid::Ptr tmp_grid_;
	OccupancyGrid::Ptr occ_grid_;

	// Publisher
	ros::Publisher cloud_filtered_pub_;
	ros::Publisher cloud_ground_plane_inliers_pub_;
	ros::Publisher cloud_ground_plane_outliers_pub_;
	ros::Publisher cloud_ground_pub_;
	ros::Publisher cloud_elevated_pub_;
	ros::Publisher tmp_occupancy_pub_;
	ros::Publisher grid_occupancy_pub_;
	ros::Publisher point_pub_;
	ros::Publisher fix_point_pub_;

	// Subscriber
	Subscriber<PointCloud2> cloud_sub_;
	
	tf2_ros::Buffer buffer_;
	tf2_ros::TransformListener tf_listener_;
	ros::Time stamp_;
	
	// Class functions
	void processPointCloud(const PointCloud2::ConstPtr & cloud);

	void fromLocalOgmToFinalOgm(const int local_grid_x, 
		const int local_grid_y, int & final_grid_index);
	
	void calculateTransMatrix();

	void fixedPoint(const int grid_x, const int grid_y);

	// Conversion functions
	void fromVeloCoordsToPolarInfo(const float x, const float y, 
	int & seg, int & bin, float & mag);
	
	void fromVeloCoordsToPolarCell(const float x, const float y,
		int & seg, int & bin);
	
	void fromPolarCellToVeloCoords(const int seg, const int bin,
		float & x, float & y);
	
	void fromLocalGridToFinalCartesian(const int grid_x, const int grid_y,
		float & x, float & y);
	
	void fromFinalCartesianToGridIndex(const float x, const float y, 
		int & grid_index);
};

} // namespace sensor_processing

#endif // sensor_processing_H
