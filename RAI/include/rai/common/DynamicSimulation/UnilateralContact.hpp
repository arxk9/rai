//
// Created by jhwangbo on 03.03.17.
//

#ifndef RAI_UNILATERALCONTACT_HPP
#define RAI_UNILATERALCONTACT_HPP

#include "Contact.hpp"

namespace rai {
namespace Dynamics {

class UnilateralContact : public Contact {
 public:
  UnilateralContact(double muIn) {
    mu = muIn;
    mu2 = muIn * muIn;
    negMu = -mu;
  }

  double mu;
  double mu2;
  double negMu;
};

}
}

#endif //RAI_UNILATERALCONTACT_HPP
