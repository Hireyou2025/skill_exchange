// tracker.js
// JavaScript implementation of AI Interview Monitoring System: Explainability Layer

class BlinkAnalysis {
    constructor() {
        this.previousClosed = false;
        this.blinkCount = 0;
    }

    _eyelidDistance(topLm, bottomLm, w, h) {
        const x1 = topLm.x * w;
        const y1 = topLm.y * h;
        const x2 = bottomLm.x * w;
        const y2 = bottomLm.y * h;
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    }

    process(landmarks, w, h) {
        if (!landmarks || !landmarks[159] || !landmarks[145] || !landmarks[386] || !landmarks[374]) return this.blinkCount;

        const leftEyeDist = this._eyelidDistance(landmarks[159], landmarks[145], w, h);
        const rightEyeDist = this._eyelidDistance(landmarks[386], landmarks[374], w, h);
        const avgDistance = (leftEyeDist + rightEyeDist) / 2;

        const closed = avgDistance < 10.0;

        if (closed && !this.previousClosed) {
            this.blinkCount++;
        }

        this.previousClosed = closed;
        return this.blinkCount;
    }
}

class GazeTracking {
    constructor() {
        this.previousDirection = "Center";
    }

    process(landmarks, w, h) {
        if (!landmarks || !landmarks[468] || !landmarks[33] || !landmarks[133]) return this.previousDirection;

        const leftIris = landmarks[468];
        const eyeLeftCorner = landmarks[33];
        const eyeRightCorner = landmarks[133];

        const irX = leftIris.x * w;
        const lx = eyeLeftCorner.x * w;
        const rx = eyeRightCorner.x * w;

        if (rx - lx === 0) {
            return this.previousDirection;
        }

        const ratio = (irX - lx) / (rx - lx);
        let currDir = "Center";

        if (ratio < 0.35) {
            currDir = "Left";
        } else if (ratio > 0.50) {
            currDir = "Right";
        } else {
            currDir = "Center";
        }

        this.previousDirection = currDir;
        return currDir;
    }
}

class HeadPoseEstimation {
    constructor() {
        this.previousPose = "Center";
    }

    process(landmarks) {
        if (!landmarks || !landmarks[1] || !landmarks[152] || !landmarks[234] || !landmarks[454] || !landmarks[10]) return this.previousPose;

        // Landmarks used:
        // 1: Nose tip
        // 152: Chin
        // 234: Left face edge
        // 454: Right face edge
        // 10: Forehead
        const nose = landmarks[1];
        const chin = landmarks[152];
        const leftEdge = landmarks[234];
        const rightEdge = landmarks[454];
        const forehead = landmarks[10];

        // Horizontal ratio (Yaw)
        const dLeft = nose.x - leftEdge.x;
        const dRight = rightEdge.x - nose.x;
        const yaw = (dLeft - dRight) / (dLeft + dRight);

        // Vertical ratio (Pitch)
        const dTop = nose.y - forehead.y;
        const dChin = chin.y - nose.y;
        const pitch = (dTop - dChin) / (dTop + dChin);

        let pose = "Center";
        if (yaw < -0.16) {
            pose = "Left";
        } else if (yaw > 0.16) {
            pose = "Right";
        } else if (pitch > 0.14) {
            pose = "Down";
        } else if (pitch < -0.18) {
            pose = "Up";
        }

        this.previousPose = pose;
        return pose;
    }
}

class IntegrityMonitoring {
    constructor() {
        this.absenceFrames = 0;
        this.absenceThreshold = 30; // ~1 second at 30 fps
    }

    process(landmarks, faceCount) {
        const flags = [];

        if (faceCount === 0 || !landmarks) {
            this.absenceFrames++;
            if (this.absenceFrames > this.absenceThreshold) {
                flags.push("Face Absent!");
            }
        } else {
            this.absenceFrames = 0;
        }

        if (faceCount > 1) {
            flags.push("Multiple Faces Detected!");
        }

        return flags;
    }
}

class EmotionInference {
    _getDistance(p1, p2, w, h) {
        const x1 = p1.x * w;
        const y1 = p1.y * h;
        const x2 = p2.x * w;
        const y2 = p2.y * h;
        return Math.hypot(x2 - x1, y2 - y1);
    }

    process(blinkCount, gazeDir, headPose, isSpeaking, landmarks = null, w = 0, h = 0) {
        let attention = 0.0;
        let confidence = 0.0;
        let focus = 0.0;
        let stress = 0.0;
        const speechEnergy = isSpeaking ? 1.0 : 0.0;
        let feeling = "Neutral";

        if (gazeDir === "Center") {
            attention += 1.0;
            confidence += 0.3;
        } else {
            attention += 0.2;
        }

        if (headPose === "Center") {
            focus += 1.0;
            confidence += 0.3;
        } else if (headPose === "Down") {
            focus += 0.2;
        } else {
            focus += 0.5;
            confidence += 0.1;
        }

        if (isSpeaking) {
            confidence += 0.2;
            attention += 0.1;
        }

        if (blinkCount > 20) {
            stress += 0.4;
            confidence -= 0.1;
        } else if (blinkCount > 10) {
            stress += 0.1;
        }

        if (landmarks && w > 0 && h > 0) {
            if (!landmarks[13] || !landmarks[14] || !landmarks[61] || !landmarks[291] || !landmarks[105] || !landmarks[159] || !landmarks[234] || !landmarks[454]) {
                // Skip emotion calculation if landmarks are missing
            } else {
                const mouthOpenDist = this._getDistance(landmarks[13], landmarks[14], w, h);
                const mouthWidthDist = this._getDistance(landmarks[61], landmarks[291], w, h);
                const browDist = this._getDistance(landmarks[105], landmarks[159], w, h);
                const faceWidth = this._getDistance(landmarks[234], landmarks[454], w, h);

                if (faceWidth > 0) {
                    const mar = mouthOpenDist / faceWidth;
                    const mouthSpread = mouthWidthDist / faceWidth;
                    const browRaise = browDist / faceWidth;

                    if (mar > 0.12 && browRaise > 0.09) {
                        feeling = "Scared / Surprised";
                        stress += 0.2;
                        confidence -= 0.1;
                    } else if (mouthSpread > 0.35 && mar < 0.15) {
                        feeling = "Happy";
                        confidence += 0.2;
                        stress -= 0.1;
                    } else if (browRaise < 0.04) {
                        feeling = "Angry / Stressed";
                        stress += 0.15;
                        confidence -= 0.1;
                    }
                }
            }
        }

        attention = Math.min(Math.max(attention, 0.0), 1.0);
        confidence = Math.min(Math.max(confidence, 0.0), 1.0);
        focus = Math.min(Math.max(focus, 0.0), 1.0);
        stress = Math.min(Math.max(stress, 0.0), 1.0);

        return {
            attention: parseFloat(attention.toFixed(2)),
            confidence: parseFloat(confidence.toFixed(2)),
            focus: parseFloat(focus.toFixed(2)),
            speech_energy: parseFloat(speechEnergy.toFixed(2)),
            stress: parseFloat(stress.toFixed(2)),
            feeling: feeling
        };
    }
}

class ScoringTrace {
    static calculateSessionTrace(sessionConfidence, sessionAttention, sessionFocus, rollingStress, integrityScore, gazeDeviationRatio) {
        // Base positive components (maximum total 100)
        let attentionContrib = sessionAttention * 40.0;
        let poseContrib = sessionFocus * 20.0;
        let comm_contrib = sessionConfidence * 40.0;

        // Penalties
        const gazePenalty = gazeDeviationRatio * 10.0;
        const stressPenalty = rollingStress * 20.0;

        // Calculate base score before integrity
        let baseScore = attentionContrib + poseContrib + comm_contrib - gazePenalty - stressPenalty;
        baseScore = Math.max(0.0, Math.min(100.0, baseScore));

        // Integrity penalty is scaling reduction
        const integrityFactor = integrityScore / 100.0;
        const finalScore = baseScore * integrityFactor;
        const integrityPenalty = baseScore - finalScore;

        // Adjust components so that their sum equals round(finalScore)
        let attn = Math.round(attentionContrib);
        let pose = Math.round(poseContrib);
        let comm = Math.round(comm_contrib);
        let gaze = Math.round(gazePenalty);
        let stress = Math.round(stressPenalty);
        let integ = Math.round(integrityPenalty);

        const calculatedFinal = attn + pose + comm - gaze - stress - integ;
        const targetFinal = Math.round(finalScore);

        // Handle rounding adjustments to ensure mathematical equality
        const diff = targetFinal - calculatedFinal;
        if (diff !== 0) {
            if (attn >= pose && attn >= comm) {
                attn += diff;
            } else if (comm >= pose) {
                comm += diff;
            } else {
                pose += diff;
            }
        }

        const finalScoreClamped = Math.max(0, Math.min(100, targetFinal));

        const contributors = {
            "Attention Consistency": attn,
            "Stable Head Pose": pose,
            "Strong Communication": comm,
            "Frequent Gaze Deviation": -gaze,
            "Elevated Stress": -stress,
            "Integrity Penalty": -integ
        };

        return [finalScoreClamped, contributors];
    }
}

class EventReasoner {
    static calculateInferenceConfidence(envMetrics) {
        let confidence = 1.0;
        const reasons = [];

        if (!envMetrics.lighting_ok) {
            confidence -= 0.15;
            reasons.push("low lighting");
        }
        if (!envMetrics.blur_ok) {
            confidence -= 0.15;
            reasons.push("motion blur");
        }
        if (!envMetrics.landmarks_stable) {
            confidence -= 0.20;
            reasons.push("unstable landmarks");
        }
        if (!envMetrics.face_visible) {
            confidence -= 0.20;
            reasons.push("partial face visibility");
        }
        if (!envMetrics.roi_stable) {
            confidence -= 0.10;
            reasons.push("ROI instability");
        }

        confidence = Math.max(0.1, parseFloat(confidence.toFixed(2)));
        return [confidence, reasons];
    }

    reasonBehavioralEvent(eventName, recentStates, envMetrics) {
        const [baseConf, envReasons] = EventReasoner.calculateInferenceConfidence(envMetrics);
        const reasons = [];

        if (!recentStates || recentStates.length === 0) {
            return {
                event: eventName,
                confidence: baseConf,
                reason: ["insufficient state history"]
            };
        }

        const avgAttention = recentStates.reduce((acc, s) => acc + (s.attention ?? 1.0), 0) / recentStates.length;
        const avgStress = recentStates.reduce((acc, s) => acc + (s.stress ?? 0.0), 0) / recentStates.length;
        const avgConfidence = recentStates.reduce((acc, s) => acc + (s.confidence ?? 1.0), 0) / recentStates.length;

        const gazeDirs = recentStates.map(s => s.gaze).filter(Boolean);
        const headPoses = recentStates.map(s => s.head_pose).filter(Boolean);
        const blinkCounts = recentStates.map(s => s.blinks).filter(val => val !== undefined);
        const speakingStatuses = recentStates.map(s => s.speaking).filter(val => val !== undefined);

        if (eventName === "Attention Drop") {
            const centerGazeRatio = gazeDirs.filter(d => d === "Center").length / (gazeDirs.length || 1);
            if (centerGazeRatio < 0.5) {
                reasons.push("off-screen gaze");
            }
            const downPoseRatio = headPoses.filter(p => p === "Down").length / (headPoses.length || 1);
            if (downPoseRatio > 0.3) {
                reasons.push("head-down posture");
            }
            const speakRatio = speakingStatuses.filter(s => s === true).length / (speakingStatuses.length || 1);
            if (speakRatio < 0.2) {
                reasons.push("speech inactivity");
            }
            if (reasons.length === 0) {
                reasons.push("general gaze deviation");
            }
        } else if (eventName === "Stress Spike") {
            if (blinkCounts.length >= 2) {
                const blinkIncrease = blinkCounts[blinkCounts.length - 1] - blinkCounts[0];
                if (blinkIncrease > 5) {
                    reasons.push("increased blink frequency");
                }
            }

            if (speakingStatuses.length > 0) {
                const speakRatio = speakingStatuses.filter(s => s === true).length / speakingStatuses.length;
                if (speakRatio > 0.0 && speakRatio < 0.3) {
                    reasons.push("speech hesitation");
                }
            }

            if (gazeDirs.length > 0) {
                const uniqueGaze = new Set(gazeDirs).size;
                if (uniqueGaze > 2) {
                    reasons.push("gaze instability");
                }
            }
            if (reasons.length === 0) {
                reasons.push("micro-expression variations");
            }
        } else if (eventName === "Communication Confidence Improved") {
            const speakRatio = speakingStatuses.filter(s => s === true).length / (speakingStatuses.length || 1);
            if (speakRatio > 0.5) {
                reasons.push("sustained verbal response");
            }
            if (avgAttention > 0.7) {
                reasons.push("centered gaze alignment");
            }
            if (reasons.length === 0) {
                reasons.push("increased speech activity");
            }
        } else if (eventName === "Integrity Warning") {
            const faceCounts = recentStates.map(s => s.face_count).filter(val => val !== undefined);
            const maxFace = faceCounts.length > 0 ? Math.max(...faceCounts) : 1;
            const minFace = faceCounts.length > 0 ? Math.min(...faceCounts) : 1;

            if (maxFace > 1) {
                reasons.push("multiple faces detected");
            } else if (minFace === 0) {
                reasons.push("candidate absent from ROI");
            } else {
                reasons.push("suspicious off-screen gaze patterns");
            }
        } else {
            reasons.push("unspecified behavioral signal");
        }

        if (envReasons && envReasons.length > 0) {
            reasons.push(`inference adjusted due to: ${envReasons.join(", ")}`);
        }

        return {
            event: eventName,
            confidence: baseConf,
            reason: reasons
        };
    }
}

class ExplanationEngine {
    constructor() {
        this.startTime = Date.now();
        this.totalFrames = 0;
        this.gazeCenterCount = 0;
        this.gazeLeftCount = 0;
        this.gazeRightCount = 0;

        this.poseCenterCount = 0;
        this.poseDownCount = 0;
        this.poseLeftCount = 0;
        this.poseRightCount = 0;
        this.poseUpCount = 0;

        this.speakingCount = 0;

        this.offScreenStart = null;
        this.prolongedGlances = [];

        this.headDownStart = null;
        this.headDownEvents = [];

        this.activeViolations = {};
        this.completedViolations = [];

        this.lowLightingFrames = 0;
        this.motionBlurFrames = 0;
        this.unstableLandmarksFrames = 0;
        this.roiInstabilityFrames = 0;
        this.faceAbsentFrames = 0;
    }

    updateFrameStats(realtimeState, rawStats, envMetrics, timestamp) {
        this.totalFrames++;

        const gaze = rawStats.gaze || "Unknown";
        if (gaze === "Center") {
            this.gazeCenterCount++;
            if (this.offScreenStart !== null) {
                const duration = timestamp - this.offScreenStart;
                if (duration >= 1.5) {
                    this.prolongedGlances.push([this.offScreenStart, timestamp, duration]);
                }
                this.offScreenStart = null;
            }
        } else if (gaze === "Left" || gaze === "Right") {
            if (gaze === "Left") {
                this.gazeLeftCount++;
            } else {
                this.gazeRightCount++;
            }
            if (this.offScreenStart === null) {
                this.offScreenStart = timestamp;
            }
        }

        const pose = rawStats.head_pose || "Unknown";
        if (pose === "Center") {
            this.poseCenterCount++;
            if (this.headDownStart !== null) {
                const duration = timestamp - this.headDownStart;
                if (duration >= 2.0) {
                    this.headDownEvents.push([this.headDownStart, timestamp, duration]);
                }
                this.headDownStart = null;
            }
        } else if (pose === "Down") {
            this.poseDownCount++;
            if (this.headDownStart === null) {
                this.headDownStart = timestamp;
            }
        } else if (pose === "Left") {
            this.poseLeftCount++;
        } else if (pose === "Right") {
            this.poseRightCount++;
        } else if (pose === "Up") {
            this.poseUpCount++;
        }

        if (rawStats.speaking) {
            this.speakingCount++;
        }

        if (!envMetrics.lighting_ok) this.lowLightingFrames++;
        if (!envMetrics.blur_ok) this.motionBlurFrames++;
        if (!envMetrics.unstable_landmarks) this.unstableLandmarksFrames++; // fixed naming from python logic
        if (!envMetrics.roi_stable) this.roiInstabilityFrames++;
        if (!envMetrics.face_visible) this.faceAbsentFrames++;

        const flags = rawStats.integrity_flags || [];
        const checkedFlags = ["Face Absent!", "Multiple Faces Detected!"];

        for (const flag of checkedFlags) {
            if (flags.includes(flag)) {
                if (this.activeViolations[flag] === undefined) {
                    this.activeViolations[flag] = timestamp;
                }
            } else {
                if (this.activeViolations[flag] !== undefined) {
                    const startT = this.activeViolations[flag];
                    delete this.activeViolations[flag];
                    const duration = timestamp - startT;
                    const severity = duration > 5.0 ? "High" : "Medium";
                    const conf = envMetrics.landmarks_stable ? 0.9 : 0.6;
                    this.completedViolations.push({
                        type: flag,
                        start: startT,
                        duration: parseFloat(duration.toFixed(1)),
                        severity: severity,
                        confidence: conf
                    });
                }
            }
        }
    }

    endActiveViolations(totalDuration) {
        for (const [flag, startT] of Object.entries(this.activeViolations)) {
            const duration = totalDuration - startT;
            const severity = duration > 5.0 ? "High" : "Medium";
            const conf = 0.8;
            this.completedViolations.push({
                type: flag,
                start: startT,
                duration: parseFloat(duration.toFixed(1)),
                severity: severity,
                confidence: conf
            });
        }
        this.activeViolations = {};
    }

    getAttentionExplanation(score, confOfInference) {
        const total = this.totalFrames || 1;
        const gazeCenterPct = Math.round((this.gazeCenterCount / total) * 100);

        const reasons = [
            `Sustained centered gaze for ${gazeCenterPct}% of interview`,
            `${this.prolongedGlances.length} prolonged off-screen glances detected`
        ];

        const downPosePct = Math.round((this.poseDownCount / total) * 100);
        if (downPosePct > 10) {
            reasons.push(`Frequent downward head pose detected (${downPosePct}% of session)`);
        }

        if (this.prolongedGlances.length > 3) {
            reasons.push("Attention instability observed during periods of off-screen glances");
        }

        return {
            score: score,
            explanation: reasons,
            confidence: confOfInference
        };
    }

    getConfidenceExplanation(score, confOfInference) {
        const total = this.totalFrames || 1;
        const speakPct = Math.round((this.speakingCount / total) * 100);

        const reasons = [
            `Vocal response active for ${speakPct}% of the sessions`
        ];

        const poseCenterPct = Math.round((this.poseCenterCount / total) * 100);
        if (poseCenterPct > 70) {
            reasons.push(`Maintained stable centered posture (${poseCenterPct}%)`);
        } else {
            reasons.push("Frequent movement or head rotation detected");
        }

        if (this.lowLightingFrames / total > 0.3) {
            reasons.push("Behavioral confidence inference adjusted due to low lighting");
        }

        return {
            score: score,
            explanation: reasons,
            confidence: confOfInference
        };
    }

    getIntegrityExplanation(score, confOfInference) {
        const reasons = [];

        if (this.completedViolations.length === 0) {
            reasons.push("No sustained integrity anomalies detected during session");
        } else {
            for (const v of this.completedViolations) {
                const vName = v.type.includes("Absent") ? "Candidate absent from ROI" : "Multiple faces detected";
                reasons.push(
                    `${vName} for ${v.duration} seconds (Severity: ${v.severity}, Confidence: ${v.confidence})`
                );
            }
        }

        const total = this.totalFrames || 1;
        const gazeDevPct = Math.round(((this.gazeLeftCount + this.gazeRightCount) / total) * 100);
        if (gazeDevPct > 40) {
            reasons.push(`Suspicious gaze deviation frequency exceeded threshold (${gazeDevPct}%)`);
        }

        return {
            score: score,
            explanation: reasons,
            confidence: confOfInference
        };
    }

    generateRecruiterSummary(finalScore, integrityScore) {
        const total = this.totalFrames || 1;
        const gazeCenterPct = Math.round((this.gazeCenterCount / total) * 100);
        const speakPct = Math.round((this.speakingCount / total) * 100);

        const summary = [];

        if (gazeCenterPct > 75) {
            summary.push("Candidate maintained strong eye contact during most responses.");
        } else if (gazeCenterPct > 50) {
            summary.push("Candidate maintained moderate eye contact, with occasional drift.");
        } else {
            summary.push("Attention instability observed; candidate frequently looked off-screen.");
        }

        const downPosePct = Math.round((this.poseDownCount / total) * 100);
        if (downPosePct > 25) {
            summary.push("Slight attention drift or head-down posture observed (likely during coding or screen reading).");
        }

        if (this.unstableLandmarksFrames / total > 0.4) {
            summary.push("Behavioral confidence decreased temporarily due to landmark tracking instability.");
        } else {
            summary.push("Stress indicators remained within standard baseline limits for most of the interview.");
        }

        if (speakPct > 30) {
            summary.push("Communication confidence improved and was well-sustained over the interview duration.");
        } else {
            summary.push("Brief vocal responses recorded; communication was mostly passive.");
        }

        if (integrityScore > 90) {
            summary.push("No major integrity violations detected.");
        } else {
            summary.push("Sustained integrity anomalies were recorded; review flags on the session timeline.");
        }

        return summary;
    }
}

class TimelineAnalytics {
    constructor() {
        this.events = [];
        this.startTime = Date.now();
        this.lastLogTime = Date.now();

        this.lowAttentionStart = null;
        this.highStressStart = null;
        this.highCommStart = null;
        this.integrityWarningStart = {};

        this.reasoner = new EventReasoner();
        this.stateHistory = [];
        this.maxHistorySize = 200;
    }

    process(realtimeState, scores, integrityFlags, envMetrics = null, rawStats = null) {
        const currentTime = Date.now();

        if (!envMetrics) {
            envMetrics = {
                lighting_ok: true,
                blur_ok: true,
                landmarks_stable: true,
                face_visible: true,
                roi_stable: true
            };
        }

        if (!rawStats) {
            rawStats = {
                gaze: realtimeState.attention > 0.5 ? "Center" : "Left",
                head_pose: realtimeState.focus > 0.5 ? "Center" : "Down",
                speaking: realtimeState.speech_energy > 0.5,
                integrity_flags: integrityFlags
            };
        }

        const faceCount = integrityFlags.includes("Face Absent!") ? 0 : (integrityFlags.includes("Multiple Faces Detected!") ? 2 : 1);
        const stateSnapshot = {
            ...realtimeState,
            ...rawStats,
            face_count: faceCount
        };

        this.stateHistory.push(stateSnapshot);
        if (this.stateHistory.length > this.maxHistorySize) {
            this.stateHistory.shift();
        }

        const elapsedSeconds = (currentTime - this.startTime) / 1000;

        // Attention Drop Detection (sustained attention < 0.5 for >3.0s)
        if (realtimeState.attention < 0.5) {
            if (this.lowAttentionStart === null) {
                this.lowAttentionStart = currentTime;
            } else if ((currentTime - this.lowAttentionStart) > 3000) {
                const reasonData = this.reasoner.reasonBehavioralEvent("Attention Drop", [...this.stateHistory], envMetrics);
                this.logEvent("Behavior", "Attention Drop", reasonData);
                this.lowAttentionStart = null;
            }
        } else {
            this.lowAttentionStart = null;
        }

        // Stress Spike Detection (sustained stress > 0.65 for >2.0s)
        if (realtimeState.stress > 0.65) {
            if (this.highStressStart === null) {
                this.highStressStart = currentTime;
            } else if ((currentTime - this.highStressStart) > 2000) {
                const reasonData = this.reasoner.reasonBehavioralEvent("Stress Spike", [...this.stateHistory], envMetrics);
                this.logEvent("Behavior", "Stress Spike", reasonData);
                this.highStressStart = null;
            }
        } else {
            this.highStressStart = null;
        }

        // Communication Improvement Detection (sustained confidence > 0.7 & speaking for >2.0s)
        if (realtimeState.confidence > 0.7 && rawStats.speaking) {
            if (this.highCommStart === null) {
                this.highCommStart = currentTime;
            } else if ((currentTime - this.highCommStart) > 2000) {
                const reasonData = this.reasoner.reasonBehavioralEvent("Communication Confidence Improved", [...this.stateHistory], envMetrics);
                this.logEvent("Behavior", "Communication Confidence Improved", reasonData);
                this.highCommStart = null;
            }
        } else {
            this.highCommStart = null;
        }

        // Integrity Warning Detection (sustained warning flags for >2.0s)
        if (integrityFlags && integrityFlags.length > 0) {
            for (const flag of integrityFlags) {
                if (this.integrityWarningStart[flag] === undefined) {
                    this.integrityWarningStart[flag] = currentTime;
                } else if ((currentTime - this.integrityWarningStart[flag]) > 2000) {
                    const reasonData = this.reasoner.reasonBehavioralEvent("Integrity Warning", [...this.stateHistory], envMetrics);
                    const cleanFlag = flag.includes("Absent") ? "Absent from ROI" : "Multiple faces detected";
                    reasonData.reason = [cleanFlag, ...reasonData.reason];
                    this.logEvent("Integrity", flag, reasonData);
                    this.integrityWarningStart[flag] = currentTime; // Reset trigger timestamp to throttle duplicates
                }
            }
        } else {
            this.integrityWarningStart = {};
        }
    }

    logEvent(eventType, eventName, reasonData) {
        const currentTime = Date.now();
        const timestamp = (currentTime - this.startTime) / 1000;

        // Prevent duplicate logs within 5 seconds
        const recentIdentical = this.events.find(
            e => e.name === eventName && (timestamp - e.timestamp) < 5.0
        );
        if (recentIdentical) {
            return;
        }

        this.lastLogTime = currentTime;

        const minutes = Math.floor(timestamp / 60);
        const seconds = Math.floor(timestamp % 60);
        const timeStr = `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;

        const newEvent = {
            timestamp: parseFloat(timestamp.toFixed(2)),
            time_str: timeStr,
            type: eventType,
            name: eventName,
            reason_data: reasonData
        };

        this.events.push(newEvent);

        // Dispatch a custom event for the UI log console
        const customEv = new CustomEvent("timeline-event-logged", { detail: newEvent });
        window.dispatchEvent(customEv);
    }
}

class ScoringEngine {
    constructor() {
        this.history = [];
        this.sessionAverages = {
            confidence: [],
            attention: [],
            focus: [],
            stress: [],
            inference_confidence: []
        };
        this.integrityScore = 100.0;

        this.trace = new ScoringTrace();
        this.explanationEngine = new ExplanationEngine();
    }

    calculate(realtimeState, integrityFlags, envMetrics = null, rawStats = null) {
        const currentTime = Date.now();
        this.history.push([currentTime, realtimeState, integrityFlags]);

        // Keep 30 seconds rolling window history
        while (this.history.length > 0 && (currentTime - this.history[0][0]) > 30000) {
            this.history.shift();
        }

        const getRollingAvg = (seconds, key) => {
            const vals = this.history
                .filter(item => (currentTime - item[0]) <= (seconds * 1000))
                .map(item => item[1][key]);
            return vals.length > 0 ? (vals.reduce((acc, v) => acc + v, 0) / vals.length) : (realtimeState[key] || 0);
        };

        this.sessionAverages.confidence.push(realtimeState.confidence);
        this.sessionAverages.attention.push(realtimeState.attention);
        this.sessionAverages.focus.push(realtimeState.focus);
        this.sessionAverages.stress.push(realtimeState.stress);

        const sessionConfidence = this.sessionAverages.confidence.reduce((acc, v) => acc + v, 0) / this.sessionAverages.confidence.length;
        const sessionAttention = this.sessionAverages.attention.reduce((acc, v) => acc + v, 0) / this.sessionAverages.attention.length;
        const sessionFocus = this.sessionAverages.focus.reduce((acc, v) => acc + v, 0) / this.sessionAverages.focus.length;

        // Integrity Score Calculation (rolling 2 seconds window checks)
        const recentFlags = this.history
            .filter(item => (currentTime - item[0]) <= 2000)
            .map(item => item[2]);

        const absentCount = recentFlags.filter(flags => flags.includes("Face Absent!")).length;
        const multipleCount = recentFlags.filter(flags => flags.includes("Multiple Faces Detected!")).length;

        if (recentFlags.length > 0) {
            if (absentCount / recentFlags.length > 0.5) {
                this.integrityScore = Math.max(0.0, this.integrityScore - 1.0);
            }
            if (multipleCount / recentFlags.length > 0.5) {
                this.integrityScore = Math.max(0.0, this.integrityScore - 2.0);
            }
            if (absentCount === 0 && multipleCount === 0) {
                this.integrityScore = Math.min(100.0, this.integrityScore + 0.1);
            }
        }

        if (!envMetrics) {
            envMetrics = {
                lighting_ok: true,
                blur_ok: true,
                landmarks_stable: true,
                face_visible: true,
                roi_stable: true
            };
        }

        if (!rawStats) {
            rawStats = {
                gaze: realtimeState.attention > 0.5 ? "Center" : "Left",
                head_pose: realtimeState.focus > 0.5 ? "Center" : "Down",
                speaking: realtimeState.speech_energy > 0.5,
                integrity_flags: integrityFlags
            };
        }

        const sessionTS = (currentTime - this.explanationEngine.startTime) / 1000;
        this.explanationEngine.updateFrameStats(realtimeState, rawStats, envMetrics, sessionTS);

        const [inferenceConfidence, envReasons] = EventReasoner.calculateInferenceConfidence(envMetrics);
        this.sessionAverages.inference_confidence.push(inferenceConfidence);

        const totalFrames = this.explanationEngine.totalFrames || 1;
        const gazeDeviationRatio = (this.explanationEngine.gazeLeftCount + this.explanationEngine.gazeRightCount) / totalFrames;

        const rollingConfidence = getRollingAvg(10, "confidence");
        const rollingAttention = getRollingAvg(10, "attention");
        const rollingStress = getRollingAvg(10, "stress");

        const [sessionScore, contributors] = ScoringTrace.calculateSessionTrace(
            sessionConfidence,
            sessionAttention,
            sessionFocus,
            rollingStress,
            this.integrityScore,
            gazeDeviationRatio
        );

        let instantScore = realtimeState.confidence * 100 - realtimeState.stress * 20;
        instantScore = Math.max(0.0, Math.min(100.0, instantScore));

        const attentionExp = this.explanationEngine.getAttentionExplanation(
            parseFloat((sessionAttention * 100).toFixed(1)),
            inferenceConfidence
        );
        const confidenceExp = this.explanationEngine.getConfidenceExplanation(
            parseFloat((sessionConfidence * 100).toFixed(1)),
            inferenceConfidence
        );
        const integrityExp = this.explanationEngine.getIntegrityExplanation(
            parseFloat(this.integrityScore.toFixed(1)),
            inferenceConfidence
        );

        return {
            instant_score: Math.round(instantScore),
            rolling_confidence: parseFloat((rollingConfidence * 100).toFixed(1)),
            rolling_attention: parseFloat((rollingAttention * 100).toFixed(1)),
            session_score: sessionScore,
            session_confidence: parseFloat((sessionConfidence * 100).toFixed(1)),
            session_attention: parseFloat((sessionAttention * 100).toFixed(1)),
            session_focus: parseFloat((sessionFocus * 100).toFixed(1)),
            integrity_score: parseFloat(this.integrityScore.toFixed(1)),
            contributors: contributors,
            inference_confidence: inferenceConfidence,
            attention_explanation: attentionExp,
            confidence_explanation: confidenceExp,
            integrity_explanation: integrityExp
        };
    }
}
