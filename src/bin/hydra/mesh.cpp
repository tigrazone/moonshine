#include "moonshine.h"

#include "mesh.hpp"
#include "renderDelegate.hpp"
#include "instancer.hpp"
#include "material.hpp"

#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/imaging/hd/extComputationUtils.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/imaging/hd/vtBufferSource.h>

#include <optional>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens,
    (st)
    (st0)
    (normals)
);

HdMoonshineMesh::HdMoonshineMesh(SdfPath const& id, const HdMoonshineRenderParam& renderParam) : HdMesh(id) {
    _material = renderParam._defaultMaterial;
}

HdDirtyBits HdMoonshineMesh::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::DirtyPoints
        | HdChangeTracker::DirtyTransform
        | HdChangeTracker::DirtyInstancer
        | HdChangeTracker::DirtyVisibility
        | HdChangeTracker::DirtyMaterialId;
}

HdDirtyBits HdMoonshineMesh::_PropagateDirtyBits(HdDirtyBits bits) const {
    return bits;
}

void HdMoonshineMesh::_InitRepr(TfToken const& reprToken, HdDirtyBits* dirtyBits) {}

std::optional<HdInterpolation> HdMoonshineMesh::FindPrimvarInterpolation(HdSceneDelegate* sceneDelegate, TfToken name) const {
    for (size_t i = 0; i < HdInterpolationCount; i++) {
        HdInterpolation interpolation = static_cast<HdInterpolation>(i);

        const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, interpolation);

        for (const HdPrimvarDescriptor& primvar : primvarDescs)
        {
            if (primvar.name == name)
            {
                return interpolation;
            }
        }
    }

    return std::nullopt;
}

template<typename T>
constexpr HdType typeToHdType() {
    if (std::is_same_v<T, GfVec2f>) {
        return HdTypeFloatVec2;
    } else if (std::is_same_v<T, GfVec3f>) {
        return HdTypeFloatVec3;
    } else {
        TF_FATAL_ERROR("unknown type");
    }
}

template<typename T>
VtArray<T> HdMoonshineMesh::ComputePrimvar(HdSceneDelegate* sceneDelegate, VtVec3iArray const& indices, TfToken primvarName) const {
    VtArray<T> primvar;
    VtValue boxedPrimvar = sceneDelegate->Get(GetId(), primvarName);
    if (boxedPrimvar.IsHolding<VtArray<T>>()) {
        std::optional<HdInterpolation> maybe_interpolation = FindPrimvarInterpolation(sceneDelegate, primvarName);
        if (!maybe_interpolation) return primvar;
        HdInterpolation interpolation = maybe_interpolation.value();
        if (interpolation == HdInterpolationFaceVarying) {
            const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
            HdMeshUtil meshUtil(&topology, GetId());

            HdVtBufferSource buffer(primvarName, boxedPrimvar);
            VtValue res;
            meshUtil.ComputeTriangulatedFaceVaryingPrimvar(buffer.GetData(), buffer.GetNumElements(), typeToHdType<T>(), &res);
            primvar = res.Get<VtArray<T>>();
        } else if (interpolation == HdInterpolationVertex) {
            VtArray<T> indexedPrimvar = boxedPrimvar.Get<VtArray<T>>();
            for (const auto index : indices) {
                primvar.push_back(indexedPrimvar[index[0]]);
                primvar.push_back(indexedPrimvar[index[1]]);
                primvar.push_back(indexedPrimvar[index[2]]);
            }
        } else {
            TF_CODING_ERROR("Mesh %s has unknown %s primvar interpolation %s!", GetId().GetText(), primvarName.GetText(), TfEnum::GetDisplayName(interpolation).c_str());
        }
    }

    return primvar;
}

void HdMoonshineMesh::Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* hdRenderParam, HdDirtyBits* dirtyBits, TfToken const& reprToken) {
    SdfPath const& id = GetId();

    HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();
    HdMoonshineRenderParam* renderParam = static_cast<HdMoonshineRenderParam*>(hdRenderParam);
    HdMoonshine* msne = renderParam->_moonshine;

    bool mesh_changed = HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points);

    if (HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points)) {
        const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
        HdMeshUtil meshUtil(&topology,id);
        VtIntArray primitiveParams;
        VtVec3iArray indices;
        meshUtil.ComputeTriangleIndices(&indices, &primitiveParams);

        VtVec3fArray indexedPoints;

        // try to find fancy points (e.g., animated ones)
        for (size_t i = 0; i < HdInterpolationCount; i++) {
            HdInterpolation interp = static_cast<HdInterpolation>(i);
            HdExtComputationPrimvarDescriptorVector compPrimvars = sceneDelegate->GetExtComputationPrimvarDescriptors(id, interp);

            for (auto const& pv: compPrimvars) {
                if (pv.name == HdTokens->points) {
                    HdExtComputationUtils::ValueStore valueStore = HdExtComputationUtils::GetComputedPrimvarValues(compPrimvars, sceneDelegate);
                    indexedPoints = valueStore.find(pv.name)->second.Get<VtVec3fArray>();
                }
            }
        }

        // no fancy points -- just use basic ones
        if (indexedPoints.size() == 0) {
            indexedPoints = sceneDelegate->Get(id, HdTokens->points).Get<VtVec3fArray>();
        }

        if (indexedPoints.size() == 0) {
            TF_CODING_ERROR("don't know what to do with empty mesh %s", id.GetText());
            return;
        }

        // hydra moonshine uses unindexed everything to support stuff like faceVarying
        VtVec3fArray points;
        for (const auto index : indices) {
            points.push_back(indexedPoints[index[0]]);
            points.push_back(indexedPoints[index[1]]);
            points.push_back(indexedPoints[index[2]]);
        }

        VtVec2fArray texcoords;
        {
            // there's some way to infer this properly but this works most of the time
            const TfToken maybeTexcoordNames[] = {
                _tokens->st,
                _tokens->st0,
            };
            TfToken texcoordName;
            for (const TfToken& name : maybeTexcoordNames)
            {
                if (FindPrimvarInterpolation(sceneDelegate, name))
                {
                    texcoordName = name;
                    break;
                }
            }

            if (!texcoordName.IsEmpty()) {
                texcoords = ComputePrimvar<GfVec2f>(sceneDelegate, indices, texcoordName);
            }
        }

        VtVec3fArray normals = ComputePrimvar<GfVec3f>(sceneDelegate, indices, _tokens->normals);
        
        // TODO: destroy mesh
        _mesh = HdMoonshineCreateMesh(msne, reinterpret_cast<const F32x3*>(points.cdata()),  reinterpret_cast<const F32x3*>(normals.cdata()), reinterpret_cast<const F32x2*>(texcoords.cdata()), points.size());

        *dirtyBits = *dirtyBits & ~HdChangeTracker::DirtyPoints;
    }

    bool old_visibility = IsVisible();
    if (HdChangeTracker::IsVisibilityDirty(*dirtyBits, id)) {
        _UpdateVisibility(sceneDelegate, dirtyBits);
        *dirtyBits = *dirtyBits & ~HdChangeTracker::DirtyVisibility;
    }
    bool new_visibility = IsVisible();

    // TODO: what is hydra SetMaterialId for
    bool material_changed = *dirtyBits & HdChangeTracker::DirtyMaterialId;
    if (*dirtyBits & HdChangeTracker::DirtyMaterialId) {
        const SdfPath& materialId = sceneDelegate->GetMaterialId(id);
        if (materialId.IsEmpty()) {
            _material = renderParam->_defaultMaterial;
        } else {
            HdSprim* sprim = renderIndex.GetSprim(HdPrimTypeTokens->material, materialId);
            if (sprim) {
                HdMoonshineMaterial* material = static_cast<HdMoonshineMaterial*>(sprim);
                _material = material->_handle;
            }
        }
        *dirtyBits = *dirtyBits & ~HdChangeTracker::DirtyMaterialId;
    }

    bool transform_changed = HdChangeTracker::IsTransformDirty(*dirtyBits, id) || HdChangeTracker::IsInstancerDirty(*dirtyBits, id);

    if (HdChangeTracker::IsTransformDirty(*dirtyBits, id)) {
        _transform = GfMatrix4f(sceneDelegate->GetTransform(id));
        *dirtyBits = *dirtyBits & ~HdChangeTracker::DirtyTransform;
    }

    const auto instancerId = GetInstancerId();
    _UpdateInstancer(sceneDelegate, dirtyBits);
    HdInstancer::_SyncInstancerAndParents(renderIndex, instancerId);

    bool instancer_count_changed = false;

    if (HdChangeTracker::IsInstancerDirty(*dirtyBits, id)) {
        const size_t old_len = _instancesTransforms.size();
        _instancesTransforms.clear();
        if (instancerId.IsEmpty()) {
            _instancesTransforms.push_back(GfMatrix4f(1.0));
        } else {
            HdInstancer *instancer = renderIndex.GetInstancer(instancerId);
            VtMatrix4dArray instanceTransforms = static_cast<HdMoonshineInstancer*>(instancer)->ComputeInstanceTransforms(id);
            for (size_t i = 0; i < instanceTransforms.size(); i++) {
                _instancesTransforms.push_back(GfMatrix4f(instanceTransforms[i]));
            }
        }
        const size_t new_len = _instancesTransforms.size();
        bool instancer_count_changed = old_len != new_len;
        *dirtyBits = *dirtyBits & ~HdChangeTracker::DirtyInstancer;
    }

    // TODO: don't actually need to recreate everything on just a material change
    bool need_to_recreate = mesh_changed || instancer_count_changed || material_changed;
    if (need_to_recreate) {
        for (const InstanceHandle instance : _instances) {
            HdMoonshineDestroyInstance(static_cast<HdMoonshineRenderParam*>(renderParam)->_moonshine, instance);
        }
        _instances.clear();

        for (size_t i = 0; i < _instancesTransforms.size(); i++) {
            GfMatrix4f instanceTransform = _transform * _instancesTransforms[i];
            const Mat3x4 matrix = Mat3x4 {
                .x = F32x4 { .x = instanceTransform[0][0], .y = instanceTransform[1][0], .z = instanceTransform[2][0], .w = instanceTransform[3][0] },
                .y = F32x4 { .x = instanceTransform[0][1], .y = instanceTransform[1][1], .z = instanceTransform[2][1], .w = instanceTransform[3][1] },
                .z = F32x4 { .x = instanceTransform[0][2], .y = instanceTransform[1][2], .z = instanceTransform[2][2], .w = instanceTransform[3][2] },
            };
            _instances.push_back(HdMoonshineCreateInstance(msne, matrix, _mesh, _material, new_visibility));
        }
    } else {
        if (transform_changed) {
            for (size_t i = 0; i < _instancesTransforms.size(); i++) {
                GfMatrix4f instanceTransform = _transform * _instancesTransforms[i];
                const Mat3x4 matrix = Mat3x4 {
                    .x = F32x4 { .x = instanceTransform[0][0], .y = instanceTransform[1][0], .z = instanceTransform[2][0], .w = instanceTransform[3][0] },
                    .y = F32x4 { .x = instanceTransform[0][1], .y = instanceTransform[1][1], .z = instanceTransform[2][1], .w = instanceTransform[3][1] },
                    .z = F32x4 { .x = instanceTransform[0][2], .y = instanceTransform[1][2], .z = instanceTransform[2][2], .w = instanceTransform[3][2] },
                };
                HdMoonshineSetInstanceTransform(msne, _instances[i], matrix);
            }
        }

        if (old_visibility != new_visibility) {
            for (const auto instance : _instances) {
                HdMoonshineSetInstanceVisibility(msne, instance, new_visibility);
            }
        }
    }

    if (!HdChangeTracker::IsClean(*dirtyBits)) {
        TF_CODING_ERROR("Dirty bits %s of %s were ignored!", HdChangeTracker::StringifyDirtyBits(*dirtyBits).c_str(), id.GetText());
    }
}

void HdMoonshineMesh::Finalize(HdRenderParam *renderParam) {
    for (const InstanceHandle instance : _instances) {
        HdMoonshineDestroyInstance(static_cast<HdMoonshineRenderParam*>(renderParam)->_moonshine, instance);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
