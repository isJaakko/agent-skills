#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

static void EmitResult(NSString *path, NSArray *candidates, NSString *errorMessage) {
    NSMutableDictionary *result = [@{
        @"path": path,
        @"candidates": candidates ?: @[]
    } mutableCopy];
    if (errorMessage != nil) {
        result[@"error"] = errorMessage;
    }
    NSError *jsonError = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:result options:0 error:&jsonError];
    if (data == nil) {
        return;
    }
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

int main(int argc, const char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "usage: vision_text_ocr ROI_X ROI_TOP_Y ROI_WIDTH ROI_HEIGHT\n");
        return 2;
    }

    const double roiX = strtod(argv[1], NULL);
    const double roiTopY = strtod(argv[2], NULL);
    const double roiWidth = strtod(argv[3], NULL);
    const double roiHeight = strtod(argv[4], NULL);
    const CGRect roi = CGRectMake(roiX, 1.0 - roiTopY - roiHeight, roiWidth, roiHeight);

    char *line = NULL;
    size_t capacity = 0;
    while (getline(&line, &capacity, stdin) != -1) {
        size_t length = strlen(line);
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
            line[--length] = '\0';
        }
        if (length == 0) {
            continue;
        }

        @autoreleasepool {
            NSString *path = [NSString stringWithUTF8String:line];
            NSURL *imageURL = [NSURL fileURLWithPath:path];
            if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
                EmitResult(path, @[], @"cannot open image");
                continue;
            }

            VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
            request.revision = VNRecognizeTextRequestRevision3;
            request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
            request.usesLanguageCorrection = NO;
            request.recognitionLanguages = @[@"en-US"];
            request.regionOfInterest = roi;

            VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithURL:imageURL options:@{}];
            NSError *visionError = nil;
            if (![handler performRequests:@[request] error:&visionError]) {
                NSString *message = visionError == nil
                    ? @"Vision request failed"
                    : [NSString stringWithFormat:@"%@ (%ld): %@", visionError.domain,
                       (long)visionError.code, visionError.localizedDescription];
                EmitResult(path, @[], message);
                continue;
            }

            NSMutableArray *candidates = [NSMutableArray array];
            for (VNRecognizedTextObservation *observation in request.results ?: @[]) {
                CGRect box = observation.boundingBox;
                for (VNRecognizedText *recognized in [observation topCandidates:3]) {
                    [candidates addObject:@{
                        @"text": recognized.string,
                        @"confidence": @(recognized.confidence),
                        @"boundingBox": @[
                            @(box.origin.x),
                            @(box.origin.y),
                            @(box.size.width),
                            @(box.size.height)
                        ]
                    }];
                }
            }
            EmitResult(path, candidates, nil);
        }
    }
    free(line);
    return 0;
}
