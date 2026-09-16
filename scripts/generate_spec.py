"""Extract the bundled arm transforms from the exact OpenArm v1 MJCF.

Only model geometry is extracted here; no IK or retargeting implementation.
Run from the repository root: python scripts/generate_spec.py
"""
from pathlib import Path
import hashlib
import io
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from openarm_teleop import model_path, spec_path


def transform(body):
    p=np.fromstring(body.get('pos','0 0 0'),sep=' ')
    q=np.fromstring(body.get('quat','1 0 0 0'),sep=' ')
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),p


def extract():
    path=model_path()
    tree=ET.parse(path)
    result={'meta/schema':'geo_kin_core.spec/1','meta/kind':'robot',
            'meta/generator_version':'openarm-reference/1','meta/source_filename':path.name,
            'meta/source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'meta/parts':['right_arm','left_arm']}
    for side in ('right','left'):
        prefix=f'{side}_arm/'
        mount=tree.find(f".//body[@name='openarm_{side}_link0']")
        r0,p0=transform(mount)
        result[prefix+'R_base_0']=r0
        result[prefix+'p_base_0']=p0
        fields={k:[] for k in ('R','p','h','joint_names','joint_lower','joint_upper')}
        for i in range(1,8):
            body=mount.find(f".//body[@name='openarm_{side}_link{i}']")
            name=f'openarm_{side}_joint{i}'
            joint=body.find(f"joint[@name='{name}']")
            r,p=transform(body)
            lower,upper=np.fromstring(joint.attrib['range'],sep=' ')
            for key,value in dict(R=r,p=p,h=np.fromstring(joint.get('axis','0 0 1'),sep=' '),
                                  joint_names=name,joint_lower=lower,joint_upper=upper).items():
                fields[key].append(value)
        result.update({prefix+k:np.asarray(v) for k,v in fields.items()})
    return result


def main():
    with zipfile.ZipFile(spec_path(),'w',compression=zipfile.ZIP_STORED) as archive:
        for key,value in sorted(extract().items()):
            buffer=io.BytesIO()
            np.lib.format.write_array(buffer,np.asarray(value),allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(key+'.npy',(1980,1,1,0,0,0)),buffer.getvalue())
    print(spec_path())

if __name__=='__main__':
    main()
