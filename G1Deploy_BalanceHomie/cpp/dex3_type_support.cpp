// The bundled SDK archive predates PressSensorState.lost/reserve. Supply the
// matching properties for the updated generated headers; otherwise its old
// two-member metadata silently skips eight bytes per tactile sensor.
#include <unitree/idl/hg/PressSensorState_.hpp>

namespace org::eclipse::cyclonedds::core::cdr {
template<>
propvec& get_type_props<unitree_hg::msg::dds_::PressSensorState_>() {
  static propvec props;
  static const bool initialized=[] {
    props.emplace_back(0,0,false,bb_unset,extensibility::ext_final);
    // Same four final, non-key members as the installed SDK's generated support.
    for(uint32_t id=0;id<4;++id)
      props.emplace_back(1,id,false,bb_32_bits,extensibility::ext_final,false);
    entity_properties_t::finish(props,key_endpoint{});
    return true;
  }();
  (void)initialized;
  return props;
}
}
