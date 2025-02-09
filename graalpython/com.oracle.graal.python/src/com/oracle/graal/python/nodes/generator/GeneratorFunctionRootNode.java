/*
 * Copyright (c) 2018, 2019, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * The Universal Permissive License (UPL), Version 1.0
 *
 * Subject to the condition set forth below, permission is hereby granted to any
 * person obtaining a copy of this software, associated documentation and/or
 * data (collectively the "Software"), free of charge and under any and all
 * copyright rights in the Software, and any and all patent rights owned or
 * freely licensable by each licensor hereunder covering either (i) the
 * unmodified Software as contributed to or provided by such licensor, or (ii)
 * the Larger Works (as defined below), to deal in both
 *
 * (a) the Software, and
 *
 * (b) any piece of software and/or hardware listed in the lrgrwrks.txt file if
 * one is included with the Software each a "Larger Work" to which the Software
 * is contributed by such licensors),
 *
 * without restriction, including without limitation the rights to copy, create
 * derivative works of, display, perform, and distribute the Software and make,
 * use, sell, offer for sale, import, export, have made, and have sold the
 * Software and the Larger Work(s), and to sublicense the foregoing rights on
 * either these or other terms.
 *
 * This license is subject to the following condition:
 *
 * The above copyright notice and either this complete permission notice or at a
 * minimum a reference to the UPL must be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.oracle.graal.python.nodes.generator;

import com.oracle.graal.python.PythonLanguage;
import com.oracle.graal.python.builtins.objects.function.Signature;
import com.oracle.graal.python.builtins.objects.function.PArguments;
import com.oracle.graal.python.nodes.PClosureFunctionRootNode;
import com.oracle.graal.python.nodes.PRootNode;
import com.oracle.graal.python.nodes.frame.MaterializeFrameNode;
import com.oracle.graal.python.parser.ExecutionCellSlots;
import com.oracle.graal.python.runtime.object.PythonObjectFactory;
import com.oracle.truffle.api.RootCallTarget;
import com.oracle.truffle.api.frame.FrameDescriptor;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.nodes.RootNode;

public class GeneratorFunctionRootNode extends PClosureFunctionRootNode {
    private final RootCallTarget callTarget;
    private final FrameDescriptor frameDescriptor;
    private final int numOfActiveFlags;
    private final int numOfGeneratorBlockNode;
    private final int numOfGeneratorForNode;
    private final ExecutionCellSlots cellSlots;
    private final String name;

    @Child private PythonObjectFactory factory = PythonObjectFactory.create();
    @Child private MaterializeFrameNode materializeNode;

    public GeneratorFunctionRootNode(PythonLanguage language, RootCallTarget callTarget, String name, FrameDescriptor frameDescriptor, ExecutionCellSlots executionCellSlots, Signature signature,
                    int numOfActiveFlags, int numOfGeneratorBlockNode, int numOfGeneratorForNode) {
        super(language, frameDescriptor, executionCellSlots, signature);
        this.callTarget = callTarget;
        this.name = name;
        this.frameDescriptor = frameDescriptor;
        this.cellSlots = executionCellSlots;
        this.numOfActiveFlags = numOfActiveFlags;
        this.numOfGeneratorBlockNode = numOfGeneratorBlockNode;
        this.numOfGeneratorForNode = numOfGeneratorForNode;
    }

    @Override
    public Object execute(VirtualFrame frame) {
        // TODO 'materialize' generator frame and create locals dict eagerly
        return factory.createGenerator(getName(), callTarget, frameDescriptor, frame.getArguments(), PArguments.getClosure(frame), cellSlots, numOfActiveFlags, numOfGeneratorBlockNode,
                        numOfGeneratorForNode);
    }

    public RootNode getFunctionRootNode() {
        return callTarget.getRootNode();
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void initializeFrame(VirtualFrame frame) {
        // nothing to do
    }

    @Override
    public boolean isPythonInternal() {
        RootNode rootNode = callTarget.getRootNode();
        return rootNode instanceof PRootNode && ((PRootNode) rootNode).isPythonInternal();
    }
}
