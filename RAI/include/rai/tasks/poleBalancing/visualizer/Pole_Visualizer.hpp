
#include "RAI_graphics.hpp"
#include "obj/Mesh.hpp"
#include "rai/common/math/RAI_math.hpp"
#include "obj/Cylinder.hpp"
#include "obj/Sphere.hpp"
#include "obj/Arrow.hpp"


namespace rai {
namespace Vis {

class Pole_Visualizer {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Pole_Visualizer();

  ~Pole_Visualizer();

  void setTerrain(std::string fileName);
  void drawWorld(HomogeneousTransform &bodyPose, double action);
  Graphics::RAI_graphics* getGraphics();


 private:
  Graphics::RAI_graphics graphics;
  Graphics::Obj::Cylinder Pole;
  Graphics::Obj::Sphere Dot;
  Graphics::Obj::Sphere Dot2;

  rai::Graphics::Obj::Mesh arrow;
  HomogeneousTransform defaultPose_;
};

}
}
