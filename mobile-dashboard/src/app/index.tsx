import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  useWindowDimensions,
  Platform
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';

// Custom Minimal Theme Colors (Avoiding Overused Gradient Clichés)
const colors = {
  bg: '#0a0c10',
  card: '#11141a',
  border: '#1c212c',
  accent: '#6366f1', // Classic Indigo
  success: '#10b981', // Emerald
  warning: '#f59e0b', // Amber
  danger: '#ef4444', // Red
  textPrimary: '#f3f4f6',
  textSecondary: '#9ca3af',
  track: '#1e293b'
};

// Preset SVGs converted to Data URIs (Base64) to be platform-independent
const presets = {
  wildfire: {
    title: 'Forest Wildfire',
    text: "RT @Cal_OES: PLS SHARE: Active wildfire response, evacuation and recovery info. Flame columns spreading in California dry forests.",
    dataUri: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200"><rect width="200" height="200" fill="%23221010"/><path d="M100,25 C100,25 135,90 135,135 C135,160 118,178 100,178 C82,178 65,160 65,135 C65,90 100,25 100,25 Z" fill="%23ef4444"/><path d="M100,65 C100,65 120,110 120,140 C120,152 110,165 100,165 C90,165 80,152 80,140 C80,110 100,65 100,65 Z" fill="%23f59e0b"/></svg>'
  },
  flood: {
    title: 'Sri Lanka Flood',
    text: "Severe monsoonal floods stranded hundreds on building roofs. Humanitarian teams arriving with emergency rafts. Roads fully submerged.",
    dataUri: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200"><rect width="200" height="200" fill="%230b1b2d"/><path d="M0,130 C45,110 75,150 115,130 C155,110 175,150 215,130 L215,200 L0,200 Z" fill="%2306b6d4"/><path d="M-20,150 C25,135 55,170 95,150 C135,135 165,170 205,150 L205,200 L-20,200 Z" fill="%232563eb"/></svg>'
  },
  earthquake: {
    title: 'Mexico Earthquake',
    text: "Devastating 7.1 magnitude earthquake caused structural collapses. Local volunteers helping search rescue squads through concrete dust and rubble.",
    dataUri: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200"><rect width="200" height="200" fill="%23251c14"/><path d="M40,60 L95,55 L85,160 L40,160 Z" fill="%2378350f"/><path d="M160,60 L96,62 L86,160 L160,160 Z" fill="%23451a03"/></svg>'
  }
};

export default function HomeScreen() {
  const { width } = useWindowDimensions();
  const isDesktop = width > 768;

  const [tweet, setTweet] = useState(presets.wildfire.text);
  const [imageUri, setImageUri] = useState(presets.wildfire.dataUri);
  const [serverIp, setServerIp] = useState('http://127.0.0.1:8000');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);

  // File Picker
  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      alert("Permission to access camera roll is required.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.8
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri);
      setPredictions(null);
    }
  };

  // Preset Selection
  const applyPreset = (key) => {
    const preset = presets[key];
    setTweet(preset.text);
    setImageUri(preset.dataUri);
    setPredictions(null);
  };

  // Inference Execution
  const analyzePost = async () => {
    if (!imageUri) {
      alert("Please select or upload a disaster image.");
      return;
    }

    setLoading(true);
    setPredictions(null);

    const formData = new FormData();
    formData.append('tweet', tweet);

    // Format file payload for cross-platform upload
    if (imageUri.startsWith('data:')) {
      // SVGs or Base64 datauris: convert to basic mock upload properties
      const filename = 'sample_preset.svg';
      formData.append('image', {
        uri: imageUri,
        type: 'image/svg+xml',
        name: filename
      } as any);
    } else {
      const filename = imageUri.split('/').pop() || 'photo.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : `image/jpeg`;
      formData.append('image', {
        uri: imageUri,
        type: type,
        name: filename
      } as any);
    }

    try {
      const response = await fetch(`${serverIp}/predict`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json'
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      setPredictions(data);
    } catch (error) {
      alert(`Connection failed: ${error.message}\nMake sure the FastAPI server is running at ${serverIp}`);
    } finally {
      setLoading(false);
    }
  };

  const renderProgressRow = (name, val) => {
    const pct = parseFloat(val);
    const color = name.toLowerCase().includes('severe') || name.toLowerCase().includes('not_inf')
      ? colors.danger 
      : name.toLowerCase().includes('mild') 
      ? colors.warning 
      : colors.success;

    return (
      <View key={name} style={styles.barRow}>
        <Text style={styles.barLabel}>{name}</Text>
        <View style={styles.barTrack}>
          <View style={[styles.barFill, { width: `${pct}%`, backgroundColor: color }]} />
        </View>
        <Text style={styles.barPct}>{pct}%</Text>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false}>
        
        {/* Editorial Minimal Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Multimodal Disaster Classification</Text>
          <Text style={styles.subtitle}>Unified Middle Fusion Visual-Text Inference Engine</Text>
        </View>

        {/* Responsive Grid */}
        <View style={[styles.gridContainer, isDesktop ? styles.desktopRow : styles.mobileCol]}>
          
          {/* Inputs Section */}
          <View style={[styles.cardContainer, isDesktop && { flex: 1.1 }]}>
            <Text style={styles.cardHeader}>Engine Inputs</Text>
            
            {/* Tweet Caption */}
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Social Media Caption</Text>
              <TextInput
                style={styles.textInput}
                multiline
                numberOfLines={3}
                placeholder="Enter social media caption..."
                placeholderTextColor={colors.textSecondary}
                value={tweet}
                onChangeText={(text) => {
                  setTweet(text);
                  setPredictions(null);
                }}
              />
            </View>

            {/* Image Drop Zone */}
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Incident Image</Text>
              <TouchableOpacity style={styles.imageZone} onPress={pickImage}>
                {imageUri ? (
                  <Image source={{ uri: imageUri }} style={styles.selectedImage} resizeMode="contain" />
                ) : (
                  <View style={styles.imagePlaceholder}>
                    <Text style={styles.placeholderText}>Click to Select Image</Text>
                  </View>
                )}
              </TouchableOpacity>
            </View>

            {/* Presets */}
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Preset Scenarios</Text>
              <View style={styles.presetsGrid}>
                {Object.keys(presets).map((key) => (
                  <TouchableOpacity key={key} style={styles.presetBtn} onPress={() => applyPreset(key)}>
                    <Text style={styles.presetBtnText}>{presets[key].title}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Actions */}
            <TouchableOpacity style={styles.actionBtn} onPress={analyzePost} disabled={loading}>
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.actionBtnText}>Analyze Multimodal State</Text>
              )}
            </TouchableOpacity>
          </View>

          {/* Outputs Section */}
          <View style={[styles.cardContainer, isDesktop && { flex: 0.9 }]}>
            <Text style={styles.cardHeader}>Engine Outputs</Text>
            
            {predictions ? (
              <View style={styles.resultsContainer}>
                
                {/* Loop prediction categories */}
                {Object.entries(predictions).map(([taskKey, taskData]: [string, any]) => {
                  if (taskData.error) {
                    return (
                      <View key={taskKey} style={styles.taskResultCard}>
                        <Text style={styles.taskTitle}>{taskKey.replace('_', ' ').toUpperCase()}</Text>
                        <Text style={styles.errorText}>Model weights missing</Text>
                      </View>
                    );
                  }

                  return (
                    <View key={taskKey} style={styles.taskResultCard}>
                      <View style={styles.taskMetaHeader}>
                        <Text style={styles.taskTitle}>{taskKey.replace('_', ' ').toUpperCase()}</Text>
                        <Text style={styles.taskVal}>{taskData.prediction}</Text>
                      </View>
                      
                      {/* Distribution breakdown */}
                      <View style={styles.distributionGroup}>
                        {Object.entries(taskData.distribution).map(([labelName, pct]) => 
                          renderProgressRow(labelName, pct)
                        )}
                      </View>
                    </View>
                  );
                })}
              </View>
            ) : (
              <View style={styles.noResultsContainer}>
                <Text style={styles.noResultsText}>
                  {loading ? 'Processing features on engine...' : 'Awaiting input features...'}
                </Text>
              </View>
            )}
          </View>
        </View>

        {/* Developer Server Configuration Setting */}
        <View style={styles.serverConfigContainer}>
          <Text style={styles.serverConfigLabel}>Backend Server URL:</Text>
          <TextInput
            style={styles.serverConfigInput}
            value={serverIp}
            onChangeText={setServerIp}
            placeholder="http://127.0.0.1:8000"
            placeholderTextColor={colors.textSecondary}
          />
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.bg
  },
  scrollContainer: {
    padding: 24,
    gap: 24
  },
  header: {
    alignSelf: 'stretch',
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    paddingBottom: 16,
    marginBottom: 8
  },
  title: {
    fontSize: 26,
    fontWeight: '800',
    color: colors.textPrimary,
    letterSpacing: -0.5
  },
  subtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    marginTop: 4,
    fontWeight: '300',
    textTransform: 'uppercase',
    letterSpacing: 0.5
  },
  gridContainer: {
    gap: 24
  },
  desktopRow: {
    flexDirection: 'row',
    alignItems: 'flex-start'
  },
  mobileCol: {
    flexDirection: 'column'
  },
  cardContainer: {
    backgroundColor: colors.card,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 24,
    gap: 20
  },
  cardHeader: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.textPrimary,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    paddingBottom: 12,
    marginBottom: 4
  },
  inputGroup: {
    gap: 8
  },
  inputLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5
  },
  textInput: {
    backgroundColor: colors.bg,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    color: colors.textPrimary,
    padding: 12,
    fontSize: 14,
    textAlignVertical: 'top',
    minHeight: 80
  },
  imageZone: {
    height: 180,
    backgroundColor: colors.bg,
    borderRadius: 8,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: colors.border,
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center'
  },
  selectedImage: {
    width: '100%',
    height: '100%'
  },
  imagePlaceholder: {
    justifyContent: 'center',
    alignItems: 'center'
  },
  placeholderText: {
    fontSize: 13,
    color: colors.textSecondary
  },
  presetsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8
  },
  presetBtn: {
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 50,
    paddingVertical: 8,
    paddingHorizontal: 16
  },
  presetBtnText: {
    fontSize: 12,
    color: colors.textPrimary
  },
  actionBtn: {
    backgroundColor: colors.accent,
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center'
  },
  actionBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700'
  },
  resultsContainer: {
    gap: 20
  },
  noResultsContainer: {
    minHeight: 280,
    justifyContent: 'center',
    alignItems: 'center'
  },
  noResultsText: {
    color: colors.textSecondary,
    fontSize: 13
  },
  taskResultCard: {
    backgroundColor: colors.bg,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 16,
    gap: 12
  },
  taskMetaHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  taskTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.textSecondary,
    letterSpacing: 0.5
  },
  taskVal: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.accent
  },
  errorText: {
    fontSize: 13,
    color: colors.danger,
    fontStyle: 'italic',
    marginTop: 4
  },
  distributionGroup: {
    gap: 10
  },
  barRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12
  },
  barLabel: {
    width: 100,
    fontSize: 12,
    color: colors.textSecondary,
    textTransform: 'capitalize'
  },
  barTrack: {
    flex: 1,
    height: 6,
    backgroundColor: colors.track,
    borderRadius: 10,
    overflow: 'hidden'
  },
  barFill: {
    height: '100%',
    borderRadius: 10
  },
  barPct: {
    width: 40,
    fontSize: 12,
    color: colors.textPrimary,
    fontWeight: '600',
    textAlign: 'right'
  },
  serverConfigContainer: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12
  },
  serverConfigLabel: {
    fontSize: 12,
    color: colors.textSecondary
  },
  serverConfigInput: {
    flex: 1,
    backgroundColor: colors.card,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: colors.border,
    color: colors.textPrimary,
    paddingVertical: 6,
    paddingHorizontal: 12,
    fontSize: 12
  }
});
