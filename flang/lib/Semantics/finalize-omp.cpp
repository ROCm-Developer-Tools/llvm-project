#include "finalize-omp.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/tools.h"

#include <list>
#include <map>
#include <variant>

namespace Fortran::semantics {

using namespace parser::literals;

class GatherCallRefs {
public:
  GatherCallRefs() {}

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  void Post(parser::Call &call) {
    if (std::holds_alternative<parser::Name>(std::get<0>(call.t).u))
      callNames.push_back(std::get<parser::Name>(std::get<0>(call.t).u));
  }

  std::list<parser::Name> callNames;
};

// This pass works by passing over the modules PFT and collecting all of the
// subroutines and functions as it passes through these nodes.
// When it meets a declare target node, it reads through the extended-list of
// the declare target directive's clause finding functions (the current function
// should be referred to in this declare target if a list is specified), if no
// list is specified the function the declare target is in, is considered the
// only function in the declare target list. These functions are then processed,
// with the algorithm looking for calls in these functions that are not included
// in the declare target but are implicitly declare target as they are invoked
// by a declare target region. These functions/subroutines are searched for in
// the map of subporograms (check if it exists, and allows access to go through
// it's own callees, to further find implicit functions) before being added to
// the original declare target extended-list (either by extended it to add a
// list, or just appending to the original list), now becoming directly declare
// target and can be further lowered as declare target. Care is taken not to add
// duplicate members to these lists, this also helps avoid infinite recursion
// when encountering a recursive function.
class ImplicitDeclareTargetCapture {
  using Subprograms = std::variant<parser::SubroutineSubprogram *,
      parser::FunctionSubprogram *>;
  Subprograms currentSubprogram_;
  std::map<std::string, Subprograms> subPrograms_;

  parser::Messages &messages_;

public:
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}
  ImplicitDeclareTargetCapture(SemanticsContext &context)
      : messages_{context.messages()} {}

  // Related to rewriting declare target specifiers to
  // contain functions nested within the primary declare
  // target function.
  void Post(parser::OpenMPDeclareTargetConstruct &x) {
    auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
    if (parser::OmpObjectList *
        objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
      markDeclTarForEachProgramInList(subPrograms_, *objectList);
    } else if (auto *clauseList{
                   parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
      for (auto &clause : clauseList->v) {
        if (auto *toClause{std::get_if<parser::OmpClause::To>(&clause.u)}) {
          markDeclTarForEachProgramInList(subPrograms_, toClause->v);
        } else if (auto *linkClause{
                       std::get_if<parser::OmpClause::Link>(&clause.u)}) {
          markDeclTarForEachProgramInList(subPrograms_, linkClause->v);
        }
      }

      // The default "declare target" inside of a function case, we must
      // create and generate an to extended-list, containing at minimum the
      // current function
      if (clauseList->v.empty()) {
        if (auto *name = getNameFromVariant(currentSubprogram_)) {
          std::list<parser::OmpObject> list;
          list.push_back(parser::OmpObject{
              parser::Designator{parser::DataRef{std::move(*name)}}});
          auto objList = parser::OmpObjectList{std::move(list)};
          markDeclTarForEachProgramInList(subPrograms_, objList);
          clauseList->v.push_back(parser::OmpClause::To{std::move(objList)});
        }
      }
    }
  }

  bool Pre(parser::FunctionSubprogram &x) {
    parser::FunctionStmt &Stmt = std::get<0>(x.t).statement;
    auto name = std::get<parser::Name>(Stmt.t);
    subPrograms_[name.ToString()] = &x;
    currentSubprogram_ = &x;
    return true;
  }

  bool Pre(parser::SubroutineSubprogram &x) {
    parser::SubroutineStmt &Stmt = std::get<0>(x.t).statement;
    auto name = std::get<parser::Name>(Stmt.t);
    subPrograms_[name.ToString()] = &x;
    currentSubprogram_ = &x;
    return true;
  }

  parser::Name *getNameFromVariant(Subprograms &x) {
    if (std::holds_alternative<parser::FunctionSubprogram *>(x)) {
      parser::FunctionStmt &Stmt =
          std::get<0>(std::get<parser::FunctionSubprogram *>(x)->t).statement;
      return &std::get<parser::Name>(Stmt.t);
    }

    if (std::holds_alternative<parser::SubroutineSubprogram *>(x)) {
      parser::SubroutineStmt &Stmt =
          std::get<0>(std::get<parser::SubroutineSubprogram *>(x)->t).statement;
      return &std::get<parser::Name>(Stmt.t);
    }

    return nullptr;
  }

  void markDeclTarForEachProgramInList(
      std::map<std::string, Subprograms> &subPrograms,
      parser::OmpObjectList &objList) {
    auto existsInList = [](parser::OmpObjectList &objList, parser::Name name) {
      for (auto &ompObject : objList.v)
        if (auto *objName{parser::Unwrap<parser::Name>(ompObject)})
          if (objName->ToString() == name.ToString())
            return true;
      return false;
    };

    GatherCallRefs gatherer{};
    for (auto &ompObject : objList.v) {
      if (auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        auto subProgram = subPrograms.find(name->ToString());
        // something other than a subroutine or function, skip it
        if (subProgram == subPrograms.end())
          continue;

        if (std::holds_alternative<parser::SubroutineSubprogram *>(
                subProgram->second))
          parser::Walk(
              *std::get<parser::SubroutineSubprogram *>(subProgram->second),
              gatherer);
        else
          parser::Walk(
              *std::get<parser::FunctionSubprogram *>(subProgram->second),
              gatherer);

        // Currently using the Function Name rather than the CallRef name,
        // unsure if these are interchangeable. However, ideally functions
        // and subroutines should probably be parser::PorcedureDesignator's
        // rather than parser::Designator's, but regular designators seem
        // to be all that is utilised in the PFT definition for OmpObjects.
        for (auto v : gatherer.callNames) {
          if (!existsInList(objList, v)) {
            objList.v.push_back(parser::OmpObject{parser::Designator{
                parser::DataRef{std::move(*getNameFromVariant(
                    subPrograms.find(v.ToString())->second))}}});
          }
        }

        gatherer.callNames.clear();
      }
    }
  }
};

bool FinalizeOMP(SemanticsContext &context, parser::Program &program) {
  ImplicitDeclareTargetCapture impCap{context};
  Walk(program, impCap);
  return !context.AnyFatalError();
}

} // namespace Fortran::semantics
