import React, { useState } from 'react';
import { Modal, Button, Collapse, Typography, Tag, Space } from 'antd';
import { BookOutlined } from '@ant-design/icons';

const { Panel } = Collapse;
const { Title, Paragraph } = Typography;

// 植物病害数据
const plantDiseases = {
  'Apple': {
    name: '苹果',
    diseases: {
      'Apple___Apple_scab': {
        name: '苹果黑星病',
        description: '苹果黑星病是由Venturia inaequalis真菌引起的常见苹果病害。',
        symptoms: '叶片和果实上出现橄榄色至黑色斑点。',
        treatment: '使用杀菌剂喷洒，清除落叶，选择抗病品种。',
        prevention: '保持果园通风，适时修剪，注意园区卫生。'
      },
      'Apple___Black_rot': {
        name: '苹果黑腐病',
        description: '由Botryosphaeria obtusa真菌引起的病害。',
        symptoms: '果实、叶片和枝条出现腐烂症状。',
        treatment: '修剪受感染的枝条，使用杀菌剂。',
        prevention: '加强果园管理，保持通风。'
      },
      'Apple___Cedar_apple_rust': {
        name: '苹果锈病',
        description: '由Gymnosporangium juniperi-virginianae真菌引起的病害。',
        symptoms: '叶片上出现橙色或黄色斑点，后期形成锈色突起。',
        treatment: '使用杀菌剂，移除附近的杜松树。',
        prevention: '避免种植感染源植物，选择抗病品种。'
      },
      'Apple___healthy': {
        name: '健康苹果',
        description: '健康的苹果植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Blueberry': {
    name: '蓝莓',
    diseases: {
      'Blueberry___healthy': {
        name: '健康蓝莓',
        description: '健康的蓝莓植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Cherry': {
    name: '樱桃',
    diseases: {
      'Cherry_(including_sour)___Powdery_mildew': {
        name: '樱桃白粉病',
        description: '由Podosphaera clandestina真菌引起的病害。',
        symptoms: '叶片、果实和嫩枝上出现白色粉状物。',
        treatment: '使用杀菌剂，修剪受感染部位。',
        prevention: '保持通风，避免过度施氮肥。'
      },
      'Cherry_(including_sour)___healthy': {
        name: '健康樱桃',
        description: '健康的樱桃植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Corn': {
    name: '玉米',
    diseases: {
      'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        name: '玉米灰斑病',
        description: '由Cercospora zeae-maydis真菌引起的病害。',
        symptoms: '叶片上出现长方形灰褐色病斑。',
        treatment: '使用杀菌剂，轮作。',
        prevention: '选择抗病品种，适当密植。'
      },
      'Corn_(maize)___Common_rust_': {
        name: '玉米普通锈病',
        description: '由Puccinia sorghi真菌引起的病害。',
        symptoms: '叶片上出现褐色至红褐色的小突起。',
        treatment: '使用杀菌剂，及时清除病株。',
        prevention: '选择抗病品种，适时播种。'
      },
      'Corn_(maize)___Northern_Leaf_Blight': {
        name: '玉米北方叶枯病',
        description: '由Exserohilum turcicum真菌引起的病害。',
        symptoms: '叶片上出现长椭圆形灰绿色至褐色病斑。',
        treatment: '使用杀菌剂，轮作。',
        prevention: '选择抗病品种，适当密植。'
      },
      'Corn_(maize)___healthy': {
        name: '健康玉米',
        description: '健康的玉米植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Grape': {
    name: '葡萄',
    diseases: {
      'Grape___Black_rot': {
        name: '葡萄黑腐病',
        description: '由Guignardia bidwellii真菌引起的病害。',
        symptoms: '叶片上出现褐色圆形病斑，果实干缩变黑。',
        treatment: '使用杀菌剂，修剪受感染部位。',
        prevention: '保持通风，避免过度灌溉。'
      },
      'Grape___Esca_(Black_Measles)': {
        name: '葡萄黑麻病',
        description: '由多种真菌复合体引起的病害。',
        symptoms: '叶片上出现黄色和褐色斑点，果实出现小黑点。',
        treatment: '修剪受感染的枝条，使用杀菌剂。',
        prevention: '避免修剪伤口，保持植株健康。'
      },
      'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        name: '葡萄叶枯病',
        description: '由Pseudocercospora vitis真菌引起的病害。',
        symptoms: '叶片上出现不规则褐色斑点，边缘呈紫色。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '保持通风，避免过度灌溉。'
      },
      'Grape___healthy': {
        name: '健康葡萄',
        description: '健康的葡萄植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Orange': {
    name: '橙子',
    diseases: {
      'Orange___Haunglongbing_(Citrus_greening)': {
        name: '柑橘黄龙病',
        description: '由细菌Candidatus Liberibacter引起的严重病害。',
        symptoms: '叶片黄化，果实变小变形，味道苦涩。',
        treatment: '目前无有效治疗方法，需移除受感染植株。',
        prevention: '控制传播媒介（木虱），使用健康种苗。'
      }
    }
  },
  'Peach': {
    name: '桃子',
    diseases: {
      'Peach___Bacterial_spot': {
        name: '桃树细菌性斑点病',
        description: '由Xanthomonas arboricola pv. pruni细菌引起的病害。',
        symptoms: '叶片、果实和枝条上出现水渍状斑点，后变为褐色。',
        treatment: '使用铜制杀菌剂，修剪受感染部位。',
        prevention: '选择抗病品种，避免过度灌溉。'
      },
      'Peach___healthy': {
        name: '健康桃树',
        description: '健康的桃树没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Pepper': {
    name: '辣椒',
    diseases: {
      'Pepper,_bell___Bacterial_spot': {
        name: '辣椒细菌性斑点病',
        description: '由Xanthomonas campestris pv. vesicatoria细菌引起的病害。',
        symptoms: '叶片和果实上出现水渍状斑点，后变为褐色。',
        treatment: '使用铜制杀菌剂，去除受感染的植株。',
        prevention: '使用健康种子，避免过度灌溉。'
      },
      'Pepper,_bell___healthy': {
        name: '健康辣椒',
        description: '健康的辣椒植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Potato': {
    name: '马铃薯',
    diseases: {
      'Potato___Early_blight': {
        name: '马铃薯早疫病',
        description: '由Alternaria solani真菌引起的病害。',
        symptoms: '叶片上出现同心圆形褐色斑点。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '轮作，保持适当株距。'
      },
      'Potato___Late_blight': {
        name: '马铃薯晚疫病',
        description: '由致病疫霉引起的严重病害。',
        symptoms: '叶片上出现水渍状病斑，茎部和块茎也会感染。',
        treatment: '使用杀菌剂，及时清除病株。',
        prevention: '选择抗病品种，避免过度灌溉。'
      },
      'Potato___healthy': {
        name: '健康马铃薯',
        description: '健康的马铃薯植株没有任何病害症状。',
        symptoms: '叶片翠绿，生长正常，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Raspberry': {
    name: '树莓',
    diseases: {
      'Raspberry___healthy': {
        name: '健康树莓',
        description: '健康的树莓植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Soybean': {
    name: '大豆',
    diseases: {
      'Soybean___healthy': {
        name: '健康大豆',
        description: '健康的大豆植株没有任何病害症状。',
        symptoms: '叶片翠绿，生长正常，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Squash': {
    name: '南瓜',
    diseases: {
      'Squash___Powdery_mildew': {
        name: '南瓜白粉病',
        description: '由多种真菌引起的病害。',
        symptoms: '叶片上出现白色粉状物。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '保持通风，避免过度施氮肥。'
      }
    }
  },
  'Strawberry': {
    name: '草莓',
    diseases: {
      'Strawberry___Leaf_scorch': {
        name: '草莓叶焦病',
        description: '由Diplocarpon earlianum真菌引起的病害。',
        symptoms: '叶片边缘出现紫红色斑点，后期叶片干枯。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '避免过度灌溉，保持适当株距。'
      },
      'Strawberry___healthy': {
        name: '健康草莓',
        description: '健康的草莓植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  },
  'Tomato': {
    name: '番茄',
    diseases: {
      'Tomato___Bacterial_spot': {
        name: '番茄细菌性斑点病',
        description: '由Xanthomonas campestris pv. vesicatoria细菌引起的病害。',
        symptoms: '叶片、果实和茎秆上出现小水渍状斑点，后变为褐色。',
        treatment: '使用铜制杀菌剂，去除受感染的植株。',
        prevention: '使用健康种子，避免过度灌溉。'
      },
      'Tomato___Early_blight': {
        name: '番茄早疫病',
        description: '由Alternaria solani真菌引起的病害。',
        symptoms: '叶片上出现同心圆形褐色斑点。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '避免叶面浇水，保持适当株距。'
      },
      'Tomato___Late_blight': {
        name: '番茄晚疫病',
        description: '由致病疫霉引起的严重病害。',
        symptoms: '叶片、茎秆和果实出现水渍状病斑。',
        treatment: '使用保护性杀菌剂，及时清除病株。',
        prevention: '选择抗病品种，注意通风和排水。'
      },
      'Tomato___Leaf_Mold': {
        name: '番茄叶霉病',
        description: '由Passalora fulva真菌引起的病害。',
        symptoms: '叶片背面出现黄色斑点，后变为橄榄绿色霉层。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '降低湿度，增加通风。'
      },
      'Tomato___Septoria_leaf_spot': {
        name: '番茄叶斑病',
        description: '由Septoria lycopersici真菌引起的病害。',
        symptoms: '叶片上出现小圆形灰白色斑点，边缘褐色。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '避免叶面浇水，轮作。'
      },
      'Tomato___Spider_mites Two-spotted_spider_mite': {
        name: '番茄二斑叶螨',
        description: '由二斑叶螨引起的害虫危害。',
        symptoms: '叶片上出现小黄点，严重时叶片干枯，出现细小蛛网。',
        treatment: '使用杀螨剂，生物防治。',
        prevention: '保持适当湿度，定期检查植株。'
      },
      'Tomato___Target_Spot': {
        name: '番茄靶斑病',
        description: '由Corynespora cassiicola真菌引起的病害。',
        symptoms: '叶片上出现同心圆形褐色斑点。',
        treatment: '使用杀菌剂，去除受感染的叶片。',
        prevention: '避免叶面浇水，保持适当株距。'
      },
      'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        name: '番茄黄化曲叶病毒病',
        description: '由番茄黄化曲叶病毒引起的病害。',
        symptoms: '叶片黄化卷曲，植株矮化，果实减少。',
        treatment: '无有效治疗方法，需移除受感染植株。',
        prevention: '控制传播媒介（烟粉虱），使用抗病品种。'
      },
      'Tomato___Tomato_mosaic_virus': {
        name: '番茄花叶病毒病',
        description: '由番茄花叶病毒引起的病害。',
        symptoms: '叶片出现黄绿相间的花叶斑驳，植株生长受阻。',
        treatment: '无有效治疗方法，需移除受感染植株。',
        prevention: '使用健康种子，避免机械传播。'
      },
      'Tomato___healthy': {
        name: '健康番茄',
        description: '健康的番茄植株没有任何病害症状。',
        symptoms: '叶片翠绿，果实饱满，无任何病斑或异常。',
        treatment: '无需处理。',
        prevention: '保持良好的园艺管理和定期检查。'
      }
    }
  }
};

interface DiseaseGuideProps {
  className?: string;
}

const DiseaseGuide: React.FC<DiseaseGuideProps> = ({ className }) => {
  const [isModalVisible, setIsModalVisible] = useState(false);

  const showModal = () => setIsModalVisible(true);
  const handleCancel = () => setIsModalVisible(false);

  return (
    <>
      <Button
        type="link"
        icon={<BookOutlined />}
        onClick={showModal}
        className={className}
      >
        支持疾病类型
      </Button>
      <Modal
        title="植物病害指南"
        open={isModalVisible}
        onCancel={handleCancel}
        width={800}
        footer={null}
      >
        <Collapse accordion>
          {Object.entries(plantDiseases).map(([plantKey, plant]) => (
            <Panel
              header={
                <Space>
                  <span>{plant.name}</span>
                  <Tag color="blue">
                    {Object.keys(plant.diseases).length}种病害
                  </Tag>
                </Space>
              }
              key={plantKey}
            >
              <Collapse>
                {Object.entries(plant.diseases).map(([diseaseKey, disease]) => (
                  <Panel
                    header={disease.name}
                    key={diseaseKey}
                  >
                    <Title level={5}>病害描述</Title>
                    <Paragraph>{disease.description}</Paragraph>
                    
                    <Title level={5}>症状特征</Title>
                    <Paragraph>{disease.symptoms}</Paragraph>
                    
                    <Title level={5}>防治方法</Title>
                    <Paragraph>{disease.treatment}</Paragraph>
                    
                    <Title level={5}>预防措施</Title>
                    <Paragraph>{disease.prevention}</Paragraph>
                  </Panel>
                ))}
              </Collapse>
            </Panel>
          ))}
        </Collapse>
      </Modal>
    </>
  );
};

export default DiseaseGuide; 