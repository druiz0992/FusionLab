// FusionLab driver for the unmodified ORB-SLAM3 tracking API.
// One read-only map-ID accessor is added by prepare_build.py for diagnostics.
#include "System.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: walking vocabulary settings frames.tsv states.csv keyframes.txt\n";
        return 1;
    }
    std::ifstream manifest(argv[3]);
    std::ofstream out(argv[4]);
    if (!manifest || !out) return 2;
    cv::FileStorage settings(argv[2], cv::FileStorage::READ);
    const int width = static_cast<int>(settings["Camera.width"]);
    const int height = static_cast<int>(settings["Camera.height"]);
    ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    out << "source_time_ns,state,map_id,processing_ms,cx,cy,cz\n";
    std::string line;
    long long origin = -1;
    auto wall_origin = std::chrono::steady_clock::now();
    while (std::getline(manifest, line)) {
        auto split = line.find('\t');
        if (split == std::string::npos) continue;
        long long stamp = std::stoll(line.substr(0, split));
        if (origin < 0) { origin = stamp; wall_origin = std::chrono::steady_clock::now(); }
        double seconds = (stamp - origin) * 1e-9;
        // Preserve source-time pacing, while processing every listed frame.
        std::this_thread::sleep_until(wall_origin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds)));
        cv::Mat image = cv::imread(line.substr(split+1), cv::IMREAD_GRAYSCALE);
        if (image.empty()) { std::cerr << "Unreadable frame\n"; return 3; }
        if (image.cols != width || image.rows != height)
            cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        auto start = std::chrono::steady_clock::now();
        Sophus::SE3f pose = slam.TrackMonocular(image, seconds);
        auto elapsed = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        int state = slam.GetTrackingState();
        out << stamp << ',' << state << ',' << slam.GetCurrentMapIdForDiagnostics() << ',' << elapsed;
        if (state == 2 || state == 5) {
            Eigen::Vector3f center = pose.inverse().translation();
            out << ',' << center.x() << ',' << center.y() << ',' << center.z();
        } else out << ",,,";
        out << '\n'; out.flush();
    }
    slam.Shutdown();
    slam.SaveKeyFrameTrajectoryTUM(argv[5]);
    return 0;
}
